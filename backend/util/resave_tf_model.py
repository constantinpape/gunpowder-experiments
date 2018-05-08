import sys
import os
import json

# try to use the tensorflow from gunpowder,
# otherwise try to revert to normal tensorflow
try:
    from gunpowder.ext import tensorflow as tf
except ImportError:
    import tensorflow as tf



# Adapted from:
# https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
def resave_model(model, model_checkpoint, model_out):

    assert os.path.exists(model)
    assert os.path.exists(model_weights + '.index')

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(model, clear_devices=True)

        # We restore the weights
        saver.restore(sess, model_checkpoint)

        # output_node_names = None
        output_node_names = 'conv_pass_0/Sigmoid'
        # serialize the node names if we don't have output node name yet
        # to check them for the output node name (usually the final activation)
        if output_node_names is None:
            s = ''
            for n in tf.get_default_graph().as_graph_def().node:
                s += str(n)
                s += '\n'
            with open('output_names', 'w') as f:
                f.write(s)
            sys.exit(0)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(model_out, "wb") as f:
            f.write(output_graph_def.SerializeToString())



if __name__ == '__main__':
    model = sys.argv[1]
    model_weights = sys.argv[2]
    model_out = sys.argv[3]
    resave_model(model, model_weights, model_out)
