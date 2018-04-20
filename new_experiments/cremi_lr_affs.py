from __future__ import print_function
import os
import sys
import backend


network_dict = {'mala': {'path': 'unet_mala', 'input_size': (84, 268, 268), 'output_size': (56, 56, 56)},
                'dtu2': {'path': 'unet_dtu2', 'input_size': (43, 430, 430), 'output_size': ((23, 218, 218))}}


def train_until(network_key, max_iteration):

    data_dir = '/nrs/saalfeld/papec/cremi2.0/training_data/V1_20180419'
    artifacts_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi/sample_ABC_padded_20160501.defects.hdf'

    # add new training samples
    samples = ['sampleA.h5',
               'sampleB.h5',
               'sampleC.h5',
               'sample0.h5',
               'sample1.h5',
               'sample2.h5']
    data_paths = [os.path.join(data_dir, sample) for sample in samples]
    nhood = backend.default_anisotropic_lr()
    print()
    print("Learning affinities with nhood:")
    print(nhood)
    print()

    path_to_meta_graph = network_dict[network_key]['path']
    input_size = network_dict[network_key]['input_size']
    output_size = network_dict[network_key]['output_size']
    backend.train_lr_affinities(path_to_meta_graph,
                                data_paths,
                                artifacts_path,
                                nhood,
                                input_size,
                                output_size,
                                max_iteration)


if __name__ == '__main__':
    network_key = sys.argv[1]
    assert network_key in network_dict
    iters = int(sys.argv[2])
    train_until(network_key, iters)
