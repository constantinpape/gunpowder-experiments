import sys
from gunpowder import *
from gunpowder.caffe import *
from stupid_predict import StupidPredict
import os
import json

import numpy as np
import h5py


def normalize(data):
    if data.dtype == np.uint8:
        factor = 1. / 255
    elif data.dtype == np.float32:
        assert raw.data.min() >= 0 and raw.data.max() <= 1, \
         "Raw values are float but not in [0,1], I don't know how to normalize. Please provide a factor."
        factor = 1.
    else:
        raise RuntimeError("False input dtype")

    return data * factor


def scale_shift(data, scale, shift):
    return data * scale + shift


def zero_out_const_sections(data):
    for z in range(data.shape[0]):
        if data[z].min() == data[z].max():
            data[z] = 0
    return data


data_dir = '/groups/saalfeld/home/papec/Work/neurodata_hdd/mala_jan_original/raw'
save_folder = './prediction_blocks'


def run_stupid_inference(sample, iteration, gpu, long_range=True):

    print("Starting prediction for gpu", gpu)
    with open('./list_gpu_%i.json' % gpu) as f:
        offset_list = json.load(f)

    prototxt = './long_range_unet.prototxt' if long_range else './default_unet.prototxt'
    weights  = './net_iter_%i.caffemodel' % iteration
    assert os.path.exists(prototxt)
    assert os.path.exists(weights)

    input_shape = (84, 268, 268)
    input_size = Coordinate((84, 268, 268)) * (40, 4, 4)
    output_size = Coordinate((56, 56, 56)) * (40, 4, 4)

    chunk_request = BatchRequest()
    chunk_request.add(VolumeTypes.RAW, input_size)
    chunk_request.add(VolumeTypes.PRED_AFFINITIES, output_size)

    raw_path = os.path.join(data_dir, 'sample_%s.h5' % sample)
    print("Loading raw_data from", raw_path)
    assert os.path.exists(raw_path)

    pred = StupidPredict(prototxt,
                         weights,
                         inputs={
                             VolumeTypes.RAW: 'data'
                         },
                         outputs={
                             VolumeTypes.PRED_AFFINITIES: 'aff_pred'
                         },
                         use_gpu=gpu)

    for ii, off in enumerate(offset_list):
        print("Predicting block", ii, "/", len(offset_list))

        bb = tuple(slice(off[i], off[i] + input_shape[i]) for i in range(len(off)))
        with h5py.File(raw_path, 'r') as f:
            data = f['data'][bb]

        data = normalize(data)
        data = scale_shift(data, 2, -1)
        data = zero_out_const_sections(data)

        out = pred({'data': data})
        out = out['aff_pred']

        save_name = 'block_%s.h5' % '_'.join([str(o) for o in off])
        save_file = os.path.join(save_folder, save_name)
        with h5py.File(save_file, 'w') as f:
            ds = f.create_dataset('data', data=out, compression='gzip')


if __name__ == '__main__':
    sample = sys.argv[1]
    iteration = int(sys.argv[2])
    gpu = int(sys.argv[3])
    run_stupid_inference(sample, iteration, gpu)
