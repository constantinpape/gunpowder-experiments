import time
from concurrent import futures
import numpy as np
import h5py
from affinities import compute_fullscale_multiscale_affinities


def prepare(sample, block_sizes):
    path = '/nrs/saalfeld/papec/cremi2.0/training_data/V1_20180419/sample%s.h5' % sample
    with h5py.File(path, 'r') as f:
        labels = f['volumes/labels/neuron_ids'][:]
    ignore_label = labels[0, 0, 0]
    print("ignore label", ignore_label)

    t0 = time.time()
    with futures.ThreadPoolExecutor(len(block_sizes)) as tp:
        tasks = [tp.submit(compute_fullscale_multiscale_affinities,
                           labels, bs,
                           haveIgnoreLabel=True,
                           ignoreLabel=ignore_label)
                 for bs in block_sizes]
        res = [t.result() for t in tasks]
    print("Calculating all affinities in", time.time() - t0, "s")

    affs = np.concatenate([re[0] for re in res], axis=0)
    mask = np.concatenate([re[1] for re in res], axis=0)
    print("Shape:")
    print(affs.shape, mask.shape)
    print(mask.dtype)

    with h5py.File(path) as f:
        ds = f.create_dataset('volumes/affinities/multiscale_affinities',
                              data=affs,
                              compression='gzip')
        ds.attrs['offset'] = [0., 0., 0.]
        ds.attrs['resolution'] = [40., 4., 4.]

        # TODO change dtype ?
        ds = f.create_dataset('volumes/affinities/mask',
                              data=mask.astype('uint8'),
                              compression='gzip')
        ds.attrs['offset'] = [0., 0., 0.]
        ds.attrs['resolution'] = [40., 4., 4.]


if __name__ == '__main__':
    sample = 'A'
    block_sizes = [[1, 3, 3], [1, 9, 9],
                   [2, 19, 19], [4, 39, 39]]
    samples = ['B', 'C', '0', '1', '2']
    with futures.ProcessPoolExecutor(5) as tp:
        tasks = [tp.submit(prepare, sample, block_sizes)
                 for sample in samples]
        [t.result() for t in tasks]
