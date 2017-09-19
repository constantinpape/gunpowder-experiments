import sys
from gunpowder import *
from gunpowder.caffe import *
import malis
import os

data_dir = '/groups/saalfeld/home/papec/Work/neurodata_hdd/mala_jan_original/raw/'

def predict_affinities(iteration, sample, gpu, long_range=True):

    prototxt = './long_range_unet.prototxt' if long_range else './default_unet.prototxt'
    weights  = './net_iter_%i.caffemodel' % iteration
    assert os.path.exists(prototxt)
    assert os.path.exists(weights)

    input_size = Coordinate((84, 268, 268)) * (40, 4, 4)
    output_size = Coordinate((56, 56, 56)) * (40, 4, 4)
    context = (input_size - output_size)/2

    chunk_request = BatchRequest()
    chunk_request.add(VolumeTypes.RAW, input_size)
    chunk_request.add(VolumeTypes.PRED_AFFINITIES, output_size)

    source = (
        Hdf5Source(
            os.path.join(data_dir, 'sample_%s.h5' % sample),
            datasets={ VolumeTypes.RAW: 'data'}
        ) +
        Pad(
            {VolumeTypes.RAW: (120, 100, 100)}
        )
    )

    snap = Snapshot(
        {
            VolumeTypes.RAW: 'volumes/raw',
            VolumeTypes.PRED_AFFINITIES: 'volumes/affinities'
        },
        every=1,
        output_dir=os.path.join('prediction', 'long_range' if long_range else 'default'),
        output_filename='sample_%s_%i.h5' % (sample, iteration)
    )

    with build(source):
        raw_spec = source.spec[VolumeTypes.RAW]
        pred_spec = raw_spec.copy()
        pred_spec.roi = pred_spec.roi.grow(-context, -context)

    predict = Predict(
        prototxt,
        weights,
        inputs={
            VolumeTypes.RAW: 'data'
        },
        outputs={
            VolumeTypes.PRED_AFFINITIES: 'aff_pred'
        },
        volume_specs={
            VolumeTypes.PRED_AFFINITIES: VolumeSpec(roi=pred_spec.roi)
        },
        use_gpu=gpu
    )

    process_pipeline = (
        source +
        Normalize() +
        IntensityScaleShift(2, -1) +
        ZeroOutConstSections() +
        predict +
        PrintProfilingStats() +
        Chunk(chunk_request) +
        snap
    )

    with build(process_pipeline):

        spec = source.spec[VolumeTypes.RAW]
        pred_spec = spec.copy()
        pred_spec.roi = pred_spec.roi.grow(-context, -context)

        whole_request = BatchRequest({
            VolumeTypes.RAW: spec,
            VolumeTypes.PRED_AFFINITIES: pred_spec
        })

        print("Requesting " + str(whole_request) + " in chunks of " + str(chunk_request))

        process_pipeline.request_batch(whole_request)

if __name__ == "__main__":
    sample = sys.argv[1]
    iteration = int(sys.argv[2])
    gpu = int(sys.argv[3])
    set_verbose(True)
    predict_affinities(iteration, sample, gpu)
