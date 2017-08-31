from volumina_viewer import volumina_n_layer
import vigra


def view_snap(snap_path):
    raw = vigra.readHDF5(snap, 'volumes/raw')
    aff = vigra.readHDF5(snap, 'volumes/labels/affinities')
    gt = vigra.rea(snap_pathp, 'volumes/labels/neuron_ids')
    mask= vigra.rea(snap_pathp, 'volumes/labels/mask')

    volumina_n_layer([raw.astype('float32'), aff, gt, mask])


if __name__ == '__main__':
    view_snap()
