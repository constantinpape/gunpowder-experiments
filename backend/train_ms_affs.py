from __future__ import print_function
import os
import math
import json
import numpy as np

from gunpowder import Coordinate, BatchRequest, VolumeTypes, Hdf5Source, VolumeSpec
from gunpowder import Normalize, RandomLocation, Reject
from gunpowder import IntensityAugment, ElasticAugment, SimpleAugment
from gunpowder import RandomProvider, DefectAugment, SplitAndRenumberSegmentationLabels, AddGtAffinities
from gunpowder import GrowBoundary, AddMultiscaleAffinities, IntensityScaleShift, ZeroOutConstSections
from gunpowder import PreCache, PrintProfilingStats  # , BalanceLabels
from gunpowder import register_volume_type, build
from gunpowder.tensorflow import Train


# TODO enable loading from existing checkpoint
def train_ms_affinities(path_to_meta_graph,
                        data_paths,
                        artifacts_path,
                        block_shapes,
                        input_size,
                        output_sizes,
                        max_iteration):

    assert isinstance(block_shapes, np.ndarray)
    assert block_shapes.ndim == 2
    assert block_shapes.shape[1] == 3
    assert all(os.path.exists(path) for path in data_paths)
    assert os.path.exists(artifacts_path)

    # number of scales is hard-coded to 4 for now
    n_scales = 4
    assert len(output_sizes) == n_scales

    with open('net_io_names.json', 'r') as f:
        net_io_names = json.load(f)

    # register new volume type (single scale)
    register_volume_type('RAW')
    register_volume_type('ALPHA_MASK')
    register_volume_type('GT_LABELS')
    register_volume_type('GT_MASK')
    # TODO we don't balance the labels, because they are soft
    # register_volume_type('GT_SCALE')

    # register new volume type (multiple scales)
    for scale in range(n_scales):
        register_volume_type('GT_AFFINITIES_S%i' % scale)
        register_volume_type('GT_AFFINITIES_MASK_S%i' % scale)
        register_volume_type('PREDICTED_AFFS_S%i' % scale)
        register_volume_type('LOSS_GRADIENT_S%i' % scale)

    # make request with all volume types we need

    # sizes for the original unet
    # input_size = Coordinate((84, 268, 268))*(40, 4, 4)
    # output_size = Coordinate((56, 56, 56))*(40, 4, 4)

    # sizes for dtu2
    # input_size = Coordinate((43, 430, 430))*(40, 4, 4)
    # output_size = Coordinate((23, 218, 218))*(40, 4, 4)
    output_sizes = [Coordinate(output_size) * (40, 4, 4)
                    for output_size in output_sizes]

    request = BatchRequest()
    # add single scale requests
    request.add(VolumeTypes.RAW, input_size)
    request.add(VolumeTypes.GT_LABELS, output_sizes[0])
    request.add(VolumeTypes.GT_MASK, output_sizes[0])
    # request.add(VolumeTypes.GT_SCALE, output_size)

    # I can't see a straight forward way to iterate over specific volume types,
    # that's why `eval` is used as a somewhat dirty hack here
    for scale in range(n_scales):
        request.add(eval('VolumeTypes.GT_AFFINITIES_S%i' % scale), output_sizes[scale])
        request.add(eval('VolumeTypes.GT_AFFINITIES_MASK_S%i' % scale), output_sizes[scale])

    data_sources = tuple(Hdf5Source(path,
                                    datasets={VolumeTypes.RAW: 'volumes/raw',
                                              VolumeTypes.GT_LABELS: 'volumes/labels/neuron_ids',
                                              VolumeTypes.GT_MASK: 'volumes/labels/mask'},
                                    volume_specs={VolumeTypes.RAW: VolumeSpec(voxel_size=(40, 4, 4)),
                                                  VolumeTypes.GT_LABELS: VolumeSpec(voxel_size=(40, 4, 4)),
                                                  VolumeTypes.GT_MASK: VolumeSpec(interpolatable=False,
                                                                                  voxel_size=(40, 4, 4))}) +
                         Normalize() +
                         RandomLocation() +
                         Reject() for path in data_paths)

    artifact_source = (Hdf5Source(artifacts_path,
                       datasets={VolumeTypes.RAW: 'defect_sections/raw',
                                 VolumeTypes.ALPHA_MASK: 'defect_sections/mask'},
                       volume_specs={VolumeTypes.RAW: VolumeSpec(voxel_size=(40, 4, 4)),
                                     VolumeTypes.ALPHA_MASK: VolumeSpec(voxel_size=(40, 4, 4))}) +
                       RandomLocation(min_masked=0.05, mask_volume_type=VolumeTypes.ALPHA_MASK) +
                       Normalize() +
                       IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
                       ElasticAugment([4, 40, 40], [0, 2, 2], [0, math.pi/2.0], subsample=8) +
                       SimpleAugment(transpose_only_xy=True))

    # define the tensorflow dictionaries
    # TODO loss weighting
    input_dict = {net_io_names['raw']: VolumeTypes.RAW}  # , net_io_names['loss_weights']: VolumeTypes.GT_SCALE}
    # update the input dictionary with multiscale inputs
    # I can't see a straight forward way to iterate over specific volume types,
    # that's why `eval` is used as a somewhat dirty hack here
    input_dict.update({net_io_names['gt_affs_s%i' % scale]: eval('VolumeTypes.GT_AFFINITIES_S%i' % scale)
                       for scale in range(n_scales)})

    output_dict = {net_io_names['affs_s%i' % scale]: eval('VolumeTypes.PREDICTED_AFFS_S%i' % scale)}
    gradient_dict = {net_io_names['affs']: eval('VolumeTypes.LOSS_GRADIENT_S%i' % scale)}

    nhood = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype='uint32')

    train_pipeline = (data_sources +
                      RandomProvider() +
                      # first augmentations: add defect augmentations (only raw data)
                      DefectAugment(prob_missing=0.025,
                                    prob_low_contrast=0.025,
                                    prob_artifact=0.025,
                                    prob_deform=0.025,
                                    artifact_source=artifact_source,
                                    contrast_scale=0.5) +
                      # next augmentation: elastic + flips in xy
                      ElasticAugment([4, 40, 40], [0, 2, 2], [0, math.pi/2.0],
                                     prob_slip=0.05, prob_shift=0.05, max_misalign=10, subsample=8) +
                      SimpleAugment(transpose_only_xy=True) +
                      # connected componets, grow boundaries and get affinities
                      SplitAndRenumberSegmentationLabels() +
                      GrowBoundary(steps=1,  # we grow less for long range affinities
                                   only_xy=True) +
                      # we use normal affinities on scale 0
                      AddGtAffinities(nhood,
                                      gt_labels=VolumeTypes.GT_LABELS,
                                      gt_affinities=VolumeTypes.GT_AFFINITIES_S0,
                                      gt_labels_mask=VolumeTypes.GT_MASK,
                                      gt_affinities_mask=VolumeTypes.GT_AFFINITIES_MASK_S0) +
                      # add multiscale affinities for the other scales
                      AddMultiscaleAffinities(block_shapes[1],
                                              gt_labels=VolumeTypes.GT_LABELS,
                                              gt_affinities=VolumeTypes.GT_AFFINITIES_S1,
                                              gt_labels_mask=VolumeTypes.GT_MASK,
                                              gt_affinities_mask=VolumeTypes.GT_AFFINITIES_MASK_S1) +
                      AddMultiscaleAffinities(block_shapes[2],
                                              gt_labels=VolumeTypes.GT_LABELS,
                                              gt_affinities=VolumeTypes.GT_AFFINITIES_S2,
                                              gt_labels_mask=VolumeTypes.GT_MASK,
                                              gt_affinities_mask=VolumeTypes.GT_AFFINITIES_MASK_S2) +
                      AddMultiscaleAffinities(block_shapes[3],
                                              gt_labels=VolumeTypes.GT_LABELS,
                                              gt_affinities=VolumeTypes.GT_AFFINITIES_S3,
                                              gt_labels_mask=VolumeTypes.GT_MASK,
                                              gt_affinities_mask=VolumeTypes.GT_AFFINITIES_MASK_S3) +
                      # intensitiy augmentations and normalizations
                      IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
                      IntensityScaleShift(2, -1) +
                      ZeroOutConstSections() +
                      # TODO we don't balance the labels for now, because of soft multi-scale affinities
                      # either check with larissa what she does for soft targets in synapses
                      # or only balance the scale 0 affinities, which are hard
                      # balance the labels
                      # BalanceLabels(labels=VolumeTypes.GT_AFFINITIES,
                      #               scales=VolumeTypes.GT_SCALE,
                      #               mask=VolumeTypes.GT_AFFINITIES_MASK) +
                      # run the actual traing
                      PreCache(cache_size=40,
                               num_workers=10) +
                      Train(path_to_meta_graph,
                            optimizer=net_io_names['optimizer'],
                            loss=net_io_names['loss'],
                            summary=net_io_names['summary'],
                            log_dir='./log/',
                            inputs=input_dict,
                            outputs=output_dict,
                            gradients=gradient_dict) +
                      # Snapshot({VolumeTypes.RAW: 'volumes/raw',
                      #           VolumeTypes.GT_LABELS: 'volumes/labels/neuron_ids',
                      #           VolumeTypes.GT_MASK: 'volumes/labels/mask',
                      #           VolumeTypes.GT_AFFINITIES: 'volumes/labels/affinities',
                      #           VolumeTypes.PRED_AFFINITIES: 'volumes/labels/prediction'},
                      #         every=5000,
                      #         output_filename='final_it={iteration}_id={id}.hdf',
                      #         additional_request=additional_request) +
                      PrintProfilingStats(every=100))

    iterations = max_iteration
    assert iterations >= 0

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(iterations):
            b.request_batch(request)
    print("Training finished")
