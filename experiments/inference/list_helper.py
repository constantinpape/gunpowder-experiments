import h5py
import os
import json

data_dir = '/groups/saalfeld/home/papec/Work/neurodata_hdd/mala_jan_original/raw'

def get_lists(sample, n_splits):
    path = os.path.join(data_dir, 'sample_%s.h5' % sample)
    with h5py.File(path, 'r') as f:
        shape = f['data'].shape

    input_shape = (84, 268, 268)
    output_shape = (56, 56, 56)

    offset = tuple((input_shape[i] - output_shape[i]) // 2
                   for i in range(3))

    in_list = []
    for z in range(0, shape[0], output_shape[0]):
        for y in range(0, shape[1], output_shape[1]):
            for x in range(0, shape[2], output_shape[2]):
                offsets = [z, y, x]
                if any([off + input_shape[i] >= shape[i] for i, off in enumerate(offsets)]):
                    continue
                in_list.append(offsets)

    out_list = []
    n_inputs = len(in_list)
    out_list = [in_list[i::n_splits] for i in range(n_splits)]

    for ii, olist in enumerate(out_list):
        list_name = './list_gpu_%i.json' % ii
        with open(list_name, 'w') as f:
            json.dump(olist, f)


if __name__ == '__main__':
    get_lists('A+', 8)
