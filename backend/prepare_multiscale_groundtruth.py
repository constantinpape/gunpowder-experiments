from concurrent import futures
import numpy as np
import vigra
from affinities import compute_fullscale_multiscale_affinities


def prepare(sample, block_sizes):
    path = '' % sample
    labels = vigra.readHDF5(path, 'volumes/labels/neuron_ids')
    ignore_label = labels[0, 0, 0]
    print("ignore label", ignore_label)

    with futures.ThreadPoolExecutor(len(block_sizes)) as tp:
        tasks = [tp.submit(compute_fullscale_multiscale_affinities,
                           labels, bs,
                           haveIgnoreLabel=True,
                           ignoreLabel=ignore_label)
                 for bs in block_sizes]
        res = [t.result() for t in tasks]

    affs = np.concatenate([re[0] for re in res], axis=0)
    mask = np.concatenate([re[1] for re in res], axis=0)
    print("Shape:")
    print(affs.shape, mask.shape)

    vigra.writeHDF5(affs, path, 'volumes/affinities/multiscale_affinities', compression='gzip')
    vigra.writeHDF5(mask, path, 'volumes/affinities/mask', compression='gzip')


if __name__ == '__main__':
    sample = 'A'
    block_sizes = [[1, 3, 3], [1, 10, 10],
                   [2, 20, 20], [4, 40, 40]]
    prepare(sample, block_sizes)
