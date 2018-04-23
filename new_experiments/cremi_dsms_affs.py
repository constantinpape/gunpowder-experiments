from __future__ import print_function
import os
import sys
import backend

from gunpowder import set_verbose


network_dict = {'dtu2': {'path': 'unet_multiscale',
                         'input_size': (43, 430, 430),
                         'output_sizes': ((23, 218, 218),
                                          (),
                                          (),
                                          ())}}


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
    output_sizes = network_dict[network_key]['output_sizes']
    backend.train_ms_affinities(path_to_meta_graph,
                                data_paths,
                                artifacts_path,
                                nhood,
                                input_size,
                                output_sizes,
                                max_iteration)


if __name__ == '__main__':
    set_verbose(False)
    network_key = sys.argv[1]
    assert network_key in network_dict
    iters = int(sys.argv[2])
    train_until(network_key, iters)
