import sys
from gunpowder import *
from gunpowder.caffe import *
import malis
import os

data_dir = '../01_data'

def predict_affinities(setup, iteration, sample, gpu):

    prototxt = os.path.join('../02_train', setup, 'net.prototxt')
    weights  = os.path.join('../02_train', setup, 'net_iter_%d.caffemodel'%iteration)

    input_size = Coordinate((84,268,268))
    output_size = Coordinate((56,56,56))
    context = (input_size - output_size)/2

    chunk_request = BatchRequest()
    chunk_request.add_volume_request(VolumeTypes.RAW, input_size)
    chunk_request.add_volume_request(VolumeTypes.PRED_AFFINITIES, output_size)

    source = Hdf5Source(
            os.path.join(data_dir, sample + '.hdf'),
            datasets = { VolumeTypes.RAW: 'volumes/raw'}
    )

    process_pipeline = (
            source +
            Normalize() +
            Pad({ VolumeTypes.RAW: (100, 100, 100) }) +
            IntensityScaleShift(2, -1) +
            ZeroOutConstSections() +
            Predict(prototxt, weights, use_gpu=gpu) +
            PrintProfilingStats() +
            Chunk(chunk_request) +
            Snapshot(
                    every=1,
                    output_dir=os.path.join('processed', setup, '%d'%iteration),
                    output_filename=sample+'.hdf'
            )
    )

    with build(process_pipeline) as p:

        raw_roi = source.get_spec().volumes[VolumeTypes.RAW]

        whole_request = BatchRequest({
                VolumeTypes.RAW: raw_roi,
                VolumeTypes.PRED_AFFINITIES: raw_roi.grow(-context, -context)
            })

        print("Requesting " + str(whole_request) + " in chunks of " + str(chunk_request))

        p.request_batch(whole_request)

if __name__ == "__main__":
    setup = sys.argv[1]
    iteration = int(sys.argv[2])
    sample = sys.argv[3]
    gpu = int(sys.argv[4])
    set_verbose(True)
    predict_affinities(setup, iteration, sample, gpu)
