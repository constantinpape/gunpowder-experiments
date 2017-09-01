from volumina_viewer import volumina_n_layer
import h5py
import numpy as np


def view_snap(snap_path):

    with h5py.File(snap_path) as f:
        ds = f['volumes/labels/mask']
        attrs = ds.attrs
        shape = ds.shape
        offset = attrs['offset']
        resolution = attrs['resolution']
        offset = [off // resolution[i] for i, off in enumerate(offset)]
        print(offset)
        print(shape)

        bb = tuple(slice(offset[i], offset[i] + s) for i, s in enumerate(shape))

        raw = f['volumes/raw'][:]
        full_shape = raw.shape

        aff_channels = f['volumes/labels/affinities'].shape[0]
        aff_shape = full_shape + (aff_channels,)
        aff_bb = bb + (slice(None),)

        aff = np.zeros(aff_shape, dtype='float32')
        aff[aff_bb] = f['volumes/labels/affinities'][:].transpose((1, 2, 3, 0))

        gt = np.zeros(full_shape, dtype='uint32')
        gt[bb] = f['volumes/labels/neuron_ids'][:]

        mask = np.zeros(full_shape, dtype='uint32')
        mask[bb] = f['volumes/labels/mask'][:]

    volumina_n_layer(
        [raw.astype('float32'), aff, gt, mask],
        ['raw', 'affinities', 'gt', 'mask']
    )


if __name__ == '__main__':
    view_snap(
        '/home/papec/mnt/papec/Work/my_projects/nnets/gunpowder-experiments/experiments/cremi/long_range/snapshots/final_it=51_id=00000198.hdf'
        #'/home/papec/mnt/papec/Work/my_projects/nnets/gunpowder-experiments/experiments/cremi/snapshots/final_it=6251_id=00024146.hdf'
    )
