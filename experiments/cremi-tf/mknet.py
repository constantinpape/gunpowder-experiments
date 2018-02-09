import sys
import networks
import tensorflow as tf
import json

def cremi_unet_inference(name='unet_inference', sample_to_isotropy=False):
    in_shape = (88, 808, 808)
    n_channels = 12

    # These values reproduce jans network
    initial_fmaps = 12
    fmap_increase = 5
    downsample_factors = [[1, 3, 3], [1, 3, 3], [3, 3, 3]] if sample_to_isotropy else \
        [[1, 3, 3], [1, 3, 3], [1, 3, 3]]

    raw = tf.placeholder(tf.float32, shape=in_shape)
    raw_batched = tf.reshape(raw, (1, 1,) + in_shape)

    unet = networks.unet(raw_batched, initial_fmaps, fmap_increase, downsample_factors)

    affs_batched = networks.conv_pass(unet,
                                      kernel_size=1,
                                      num_fmaps=n_channels,
                                      num_repetitions=1,
                                      activation='sigmoid')

    output_shape_batched = affs_batched.get_shape().as_list()
    output_shape = output_shape_batched[1:]  # strip the batch dimension

    affs = tf.reshape(affs_batched, output_shape)

    tf.train.export_meta_graph(filename='%s.meta' % name)


def cremi_unet(name='unet', sample_to_isotropy=False):
    in_shape = (84, 268, 268)
    n_channels = 12

    # These values reproduce jans network
    initial_fmaps = 12
    fmap_increase = 5
    downsample_factors = [[1, 3, 3], [1, 3, 3], [3, 3, 3]] if sample_to_isotropy else \
        [[1, 3, 3], [1, 3, 3], [1, 3, 3]]

    raw = tf.placeholder(tf.float32, shape=in_shape)
    raw_batched = tf.reshape(raw, (1, 1,) + in_shape)

    unet = networks.unet(raw_batched, initial_fmaps, fmap_increase, downsample_factors)

    affs_batched = networks.conv_pass(unet,
                                      kernel_size=1,
                                      num_fmaps=n_channels,
                                      num_repetitions=1,
                                      activation='sigmoid')

    output_shape_batched = affs_batched.get_shape().as_list()
    output_shape = output_shape_batched[1:] # strip the batch dimension

    affs = tf.reshape(affs_batched, output_shape)

    gt_affs = tf.placeholder(tf.float32, shape=output_shape)

    loss_weights = tf.placeholder(tf.float32, shape=output_shape)

    loss = tf.losses.mean_squared_error(
        gt_affs,
        affs,
        loss_weights)
    tf.summary.scalar('loss_total', loss)

    opt = tf.train.AdamOptimizer(
        learning_rate=0.5e-4,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8)
    optimizer = opt.minimize(loss)
    #for trainable in tf.trainable_variables():
    #    networks.tf_var_summary(trainable)
    merged = tf.summary.merge_all()

    tf.train.export_meta_graph(filename='%s.meta' % name)

    names = {'raw': raw.name,
             'affs': affs.name,
             'gt_affs': gt_affs.name,
             'loss_weights': loss_weights.name,
             'loss': loss.name,
             'optimizer': optimizer.name,
             'summary': merged.name}

    with open('net_io_names.json', 'w') as f:
        json.dump(names, f)


if __name__ == "__main__":
    sample_to_isotropy = True
    cremi_unet(sample_to_isotropy=sample_to_isotropy)
    cremi_unet_inference(sample_to_isotropy=sample_to_isotropy)
