from __future__ import print_function
import os
import math
import json
import numpy as np

from gunpowder import Coordinate, BatchRequest, VolumeTypes, Hdf5Source, VolumeSpec
from gunpowder import Normalize, RandomLocation, Reject
from gunpowder import IntensityAugment, ElasticAugment, SimpleAugment
from gunpowder import RandomProvider, DefectAugment, SplitAndRenumberSegmentationLabels
from gunpowder import GrowBoundary, AddGtAffinities, IntensityScaleShift, ZeroOutConstSections
from gunpowder import BalanceLabels, PreCache, PrintProfilingStats
from gunpowder import register_volume_type, build
from gunpowder.tensorflow import Train


# TODO enable loading from existing checkpoint
def train_lr_affinities(data_paths, artifacts_path, nhood, max_iteration):

    assert isinstance(nhood, np.ndarray)
    assert nhood.ndim == 2
    assert nhood.shape[1] == 3
    assert all(os.path.exists(path) for path in data_paths)
    assert os.path.exists(artifacts_path)

    # TODO don't hardcode path !
    with open('net_io_names.json', 'r') as f:
        net_io_names = json.load(f)

    # register new volume type
    register_volume_type('RAW')
    register_volume_type('ALPHA_MASK')
    register_volume_type('GT_LABELS')
    register_volume_type('GT_MASK')
    register_volume_type('GT_SCALE')
    register_volume_type('GT_AFFINITIES')
    register_volume_type('PREDICTED_AFFS')
    register_volume_type('LOSS_GRADIENT')

    # make request with all volume types we need

    # sizes for the original unet
    # input_size = Coordinate((84, 268, 268))*(40, 4, 4)
    # output_size = Coordinate((56, 56, 56))*(40, 4, 4)

    # sizes for dtu2
    input_size = Coordinate((43, 430, 430))*(40, 4, 4)
    output_size = Coordinate((23, 218, 218))*(40, 4, 4)

    request = BatchRequest()
    request.add(VolumeTypes.RAW, input_size)
    request.add(VolumeTypes.GT_LABELS, output_size)
    request.add(VolumeTypes.GT_MASK, output_size)
    request.add(VolumeTypes.GT_AFFINITIES, output_size)
    request.add(VolumeTypes.GT_AFFINITIES_MASK, output_size)
    request.add(VolumeTypes.GT_MASK, output_size)
    request.add(VolumeTypes.GT_SCALE, output_size)

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
                      AddGtAffinities(nhood, gt_labels_mask=VolumeTypes.GT_MASK) +
                      # intensitiy augmentations and normalizations
                      IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
                      IntensityScaleShift(2, -1) +
                      ZeroOutConstSections() +
                      # balance the labels
                      BalanceLabels(labels=VolumeTypes.GT_AFFINITIES,
                                    scales=VolumeTypes.GT_SCALE,
                                    mask=VolumeTypes.GT_AFFINITIES_MASK) +
                      # run the actual traing
                      PreCache(cache_size=40,
                               num_workers=10) +
                      Train('unet',
                            optimizer=net_io_names['optimizer'],
                            loss=net_io_names['loss'],
                            summary=net_io_names['summary'],
                            log_dir='./log/',
                            inputs={net_io_names['raw']: VolumeTypes.RAW,
                                    net_io_names['gt_affs']: VolumeTypes.GT_AFFINITIES,
                                    net_io_names['loss_weights']: VolumeTypes.GT_SCALE},
                            outputs={net_io_names['affs']: VolumeTypes.PREDICTED_AFFS},
                            gradients={net_io_names['affs']: VolumeTypes.LOSS_GRADIENT}) +
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
