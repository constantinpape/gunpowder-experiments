from __future__ import print_function
import os
import sys
import backend


def train_until(max_iteration):

    data_dir = '/nrs/saalfeld/papec/cremi2.0/training_data/V1_20180419'
    artifacts_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi/sample_ABC_padded_20160501.defects.hdf'

    # TODO add new training samples
    samples = ['sampleA.h5',
               'sampleB.h5',
               'sampleC.h5']
    data_paths = [os.path.join(data_dir, sample) for sample in samples]

    nhood = backend.default_anisotropic_lr()
    print()
    print("Learning affinities with nhood:")
    print(nhood)
    print()

    backend.train_lr_affinities(data_paths, artifacts_path, nhood, max_iteration)


if __name__ == '__main__':
    iters = int(sys.argv[1])
    train_until(iters)
