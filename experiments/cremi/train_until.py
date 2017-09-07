from __future__ import print_function
import sys
from gunpowder import *
from gunpowder.caffe import *
import malis
import os
import glob
import math
import numpy as np

data_dir = '/groups/saalfeld/home/papec/Work/neurodata_hdd/mala_jan_original/raw'
samples = [
    'sample_A.h5',
    'sample_B.h5',
    'sample_C.h5'
]
phase_switch = 10000

# long range neighborhood
def make_long_range_nhood(long_range=4, xy_ranges=[1, 4, 8, 16]):
    assert len(xy_ranges) == long_range
    nhood = []
    for i in range(long_range):
        nhood_z = [- i, 0, 0]
        nhood.append(nhood_z)
        range_xy = - xy_ranges[i]
        nhood_y = [0, range_xy, 0]
        nhood.append(nhood_y)
        nhood_x = [0, 0, range_xy]
        nhood.append(nhood_x)
    return np.array(nhood, dtype='int32')


def train_until(max_iteration, gpu, long_range=False):

    # get most recent training result
    solverstates = [int(f.split('.')[0].split('_')[-1]) for f in glob.glob('net_iter_*.solverstate')]
    if len(solverstates) > 0:
        trained_until = max(solverstates)
        print("Resuming training from iteration " + str(trained_until))
    else:
        trained_until = 0
        print("Starting fresh training")

    if trained_until < phase_switch and max_iteration > phase_switch:
        # phase switch lies in-between, split training into to parts
        train_until(phase_switch, gpu)
        trained_until = phase_switch

    if max_iteration <= phase_switch:
        phase = 'euclid'
    else:
        phase = 'malis'

    net_file = 'default_unet.prototxt' if not long_range else 'long_range_unet.prototxt'
    print()
    print("Training until " + str(max_iteration) + " in phase " + phase)
    print("Training with architecture from %s" % net_file)
    print("Using gpu: %i" % gpu)
    print()

    solver_parameters = SolverParameters()
    solver_parameters.train_net = net_file
    solver_parameters.base_lr = 0.5e-4
    solver_parameters.momentum = 0.95
    solver_parameters.momentum2 = 0.999
    solver_parameters.delta = 1e-8
    solver_parameters.weight_decay = 0.000005
    solver_parameters.lr_policy = 'inv'
    solver_parameters.gamma = 0.0001
    solver_parameters.power = 0.75
    solver_parameters.snapshot = 2000
    solver_parameters.snapshot_prefix = 'net'
    solver_parameters.type = 'Adam'
    if trained_until > 0:
        solver_parameters.resume_from = 'net_iter_' + str(trained_until) + '.solverstate'
    else:
        solver_parameters.resume_from = None
    solver_parameters.train_state.add_stage(phase)

    # register new volume type
    register_volume_type('MALIS_COMP_LABEL')
    register_volume_type('LOSS_SCALE')

    # make request with all volume types we need
    request = BatchRequest()
    request.add(VolumeTypes.RAW, Coordinate((84,268,268))*(40,4,4))
    request.add(VolumeTypes.GT_LABELS, Coordinate((56,56,56))*(40,4,4))
    request.add(VolumeTypes.GT_MASK, Coordinate((56,56,56))*(40,4,4))
    request.add(VolumeTypes.GT_AFFINITIES, Coordinate((56,56,56))*(40,4,4))
    request.add(VolumeTypes.LOSS_SCALE, Coordinate((56,56,56))*(40,4,4))
    request.add(VolumeTypes.MALIS_COMP_LABEL, Coordinate((56,56,56))*(40,4,4))

    # additional gradients for snapshots TODO add loss gradient
    additional_request = BatchRequest()
    additional_request.add(VolumeTypes.PRED_AFFINITIES, Coordinate((56,56,56))*(40,4,4))

    data_sources = tuple(
        Hdf5Source(
            os.path.join(data_dir, sample),
            datasets = {
                VolumeTypes.RAW: 'data',
                VolumeTypes.GT_LABELS: 'volumes/labels/neuron_ids_notransparency',
                VolumeTypes.GT_MASK: 'volumes/labels/mask',
            }
        ) +
        Normalize() +
        RandomLocation() +
        Reject()
        for sample in samples
    )

    artifact_source = (
        Hdf5Source(
            os.path.join(data_dir, 'sample_ABC_padded_20160501.defects.hdf'),
            datasets = {
                VolumeTypes.RAW: 'defect_sections/raw',
                VolumeTypes.ALPHA_MASK: 'defect_sections/mask',
            }
        ) +
        RandomLocation(min_masked=0.05, mask_volume_type=VolumeTypes.ALPHA_MASK) +
        Normalize() +
        IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        ElasticAugment([4,40,40], [0,2,2], [0,math.pi/2.0], subsample=8) +
        SimpleAugment(transpose_only_xy=True)
    )

    train_pipeline = (
        data_sources +
        RandomProvider() +
        # first augmentations: add defect augmentations (only raw data)
        DefectAugment(
            prob_missing=0.03,
            prob_low_contrast=0.01,
            prob_artifact=0.03,
            prob_deform=0.02,
            artifact_source=artifact_source,
            contrast_scale=0.5) +
        # next augmentation: elastic + flips in xy
        ElasticAugment(
            [4,40,40], [0,2,2], [0,math.pi/2.0], prob_slip=0.05,prob_shift=0.05,max_misalign=10, subsample=8
        ) +
        SimpleAugment(transpose_only_xy=True) +
        # connected componets, grow boundaries and get affinities
        SplitAndRenumberSegmentationLabels() +
        GrowBoundary(
            steps=4 if not long_range else 2, # we grow less for long range affinities
            only_xy=True) +
        AddGtAffinities(
            malis.mknhood3d() if not long_range else make_long_range_nhood()) +
        # intensitiy augmentations and normalizations
        IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        IntensityScaleShift(2, -1) +
        ZeroOutConstSections() +
        # magic prepare malis node
        PrepareMalis() +
	# balance the labels
        BalanceLabels(labels_to_loss_scale_volume={VolumeTypes.GT_AFFINITIES: VolumeTypes.LOSS_SCALE},
                        labels_to_mask_volumes={VolumeTypes.GT_AFFINITIES: (VolumeTypes.GT_MASK,)}) +
	# run the actual traing
        PreCache(
            cache_size=40,
            num_workers=10) +
        Train(solver_parameters,
              inputs={VolumeTypes.RAW: 'data',
                      VolumeTypes.GT_AFFINITIES: 'aff_label',
                      VolumeTypes.LOSS_SCALE: 'scale',
                      VolumeTypes.MALIS_COMP_LABEL: 'comp_label',
                      'affinity_neighborhood': 'nhood'},
              outputs={VolumeTypes.PRED_AFFINITIES: 'aff_pred'},
              gradients={VolumeTypes.LOSS_GRADIENT: 'aff_pred'},
              use_gpu=gpu) +
        Snapshot(
            {
                VolumeTypes.RAW: 'volumes/raw',
                VolumeTypes.GT_LABELS: 'volumes/labels/neuron_ids',
                VolumeTypes.GT_MASK: 'volumes/labels/mask',
                VolumeTypes.GT_AFFINITIES: 'volumes/labels/affinities',
                VolumeTypes.PRED_AFFINITIES: 'volumes/labels/prediction'
            },
            every=100,
            output_filename='final_it={iteration}_id={id}.hdf',
            additional_request=additional_request) +
        PrintProfilingStats(every=100)
    )

    iterations = max_iteration - trained_until
    assert iterations >= 0

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(iterations):
            b.request_batch(request)
    print("Training finished")


if __name__ == "__main__":
    set_verbose(False)
    iteration = int(sys.argv[1])
    gpu = int(sys.argv[2])
    train_until(iteration, gpu)
