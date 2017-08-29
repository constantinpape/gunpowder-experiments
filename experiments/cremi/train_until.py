from __future__ import print_function
import sys
from gunpowder import *
from gunpowder.caffe import *
import malis
import os
import glob
import math

data_dir = '../../01_data'
samples = [
    'sample_A_padded_20160501.aligned.filled.cropped',
    'sample_B_padded_20160501.aligned.filled.cropped',
    'sample_C_padded_20160501.aligned.filled.cropped.0:90'
]
phase_switch = 10000

def train_until(max_iteration, gpu):

    # get most recent training result
    solverstates = [ int(f.split('.')[0].split('_')[-1]) for f in glob.glob('net_iter_*.solverstate') ]
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
    print("Traing until " + str(max_iteration) + " in phase " + phase)

    solver_parameters = SolverParameters()
    solver_parameters.train_net = 'net.prototxt'
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

    request = BatchRequest()
    request.add_volume_request(VolumeType.RAW, (84,268,268))
    request.add_volume_request(VolumeType.GT_LABELS, (56,56,56))
    request.add_volume_request(VolumeType.GT_MASK, (56,56,56))
    request.add_volume_request(VolumeType.GT_AFFINITIES, (56,56,56))

    data_sources = tuple(
        Hdf5Source(
            os.path.join(data_dir, sample + '.hdf'),
            datasets = {
                VolumeType.RAW: 'volumes/raw',
                VolumeType.GT_LABELS: 'volumes/labels/neuron_ids_notransparency',
                VolumeType.GT_MASK: 'volumes/labels/mask',
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
                VolumeType.RAW: 'defect_sections/raw',
                VolumeType.ALPHA_MASK: 'defect_sections/mask',
            }
        ) +
        RandomLocation(min_masked=0.05, mask_volume_type=VolumeType.ALPHA_MASK) +
        Normalize() +
        IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        ElasticAugment([4,40,40], [0,2,2], [0,math.pi/2.0], subsample=8) +
        SimpleAugment(transpose_only_xy=True)
    )

    snapshot_request = BatchRequest({VolumeType.LOSS_GRADIENT: request.volumes[VolumeType.GT_AFFINITIES]})

    train_pipeline = (
        data_sources +
        RandomProvider() +
        ElasticAugment([4,40,40], [0,2,2], [0,math.pi/2.0], prob_slip=0.05,prob_shift=0.05,max_misalign=10, subsample=8) +
        SimpleAugment(transpose_only_xy=True) +
        GrowBoundary(steps=4, only_xy=True) +
        AddGtAffinities(malis.mknhood3d()) +
        SplitAndRenumberSegmentationLabels() +
        IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        DefectAugment(
            prob_missing=0.03,
            prob_low_contrast=0.01,
            prob_artifact=0.03,
            artifact_source=artifact_source,
            contrast_scale=0.5) +
        IntensityScaleShift(2,-1) +
        ZeroOutConstSections() +
        PreCache(
            request,
            cache_size=40,
            num_workers=10) +
        Train(solver_parameters, use_gpu=gpu) +
        Snapshot(every=100, output_filename='batch_{iteration}.hdf', additional_request=snapshot_request) +
        PrintProfilingStats(every=10)
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
