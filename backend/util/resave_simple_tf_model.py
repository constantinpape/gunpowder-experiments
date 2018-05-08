import sys
import os
import json

# try to use the tensorflow from gunpowder,
# otherwise try to revert to normal tensorflow
try:
    from gunpowder.ext import tensorflow as tf
except ImportError:
    import tensorflow as tf


def resave_model(model, model_checkpoint, model_out, net_io_names):

    assert os.path.exists(model)
    assert os.path.exists(model_weights + '.index')

    with open(net_io_names, 'r') as f:
        net_io_names = json.load(f)

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(model, clear_devices=True)
        # We restore the weights
        saver.restore(sess, model_checkpoint)

        # Get the input and output tensors
        input_ = sess.graph.as_graph_element(net_io_names['raw'])
        output = sess.graph.as_graph_element(net_io_names['affs'])
        input_info = tf.saved_model.utils.build_tensor_info(input_)
        output_info = tf.saved_model.utils.build_tensor_info(output)

        # unfortunately the tensorflow version currently in
        # gunpowder does not support this
        # tf.saved_model.simple_save(session, model_out,
        #                            inputs={'raw': input_info},
        #                            outputs={'affs': output_info})

	builder = tf.saved_model.builder.SavedModelBuilder(model_out)
        signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'raw': input_info},
                outputs={'affs': output_info},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
        )
        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map={'predict_images': signature}
            )
	builder.save(as_text=True)



if __name__ == '__main__':
    model = sys.argv[1]
    model_weights = sys.argv[2]
    model_out = sys.argv[3]
    net_io_names = sys.argv[4]
    resave_model(model, model_weights, model_out, net_io_names)
