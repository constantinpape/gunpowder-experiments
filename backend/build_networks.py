from .networks import unet_dtu2, unet_mala
from .networks import conv_pass_dtu2, conv_pass_mala
import tensorflow as tf
import json


def get_default_adam():
    return tf.train.AdamOptimizer(learning_rate=0.5e-4,
                                  beta1=0.95,
                                  beta2=0.999,
                                  epsilon=1e-8)


def build_unet_mala(name='unet_mala', sample_to_isotropy=False, have_weights=True):
    in_shape = (84, 268, 268)
    n_channels = 12

    # These values reproduce jans network
    initial_fmaps = 12
    fmap_increase = 5
    downsample_factors = [[1, 3, 3], [1, 3, 3], [3, 3, 3]] if sample_to_isotropy else \
        [[1, 3, 3], [1, 3, 3], [1, 3, 3]]

    raw = tf.placeholder(tf.float32, shape=in_shape)
    raw_batched = tf.reshape(raw, (1, 1,) + in_shape)

    unet = unet_mala(raw_batched, initial_fmaps, fmap_increase, downsample_factors)

    affs_batched = conv_pass_mala(unet,
                                  kernel_size=1,
                                  num_fmaps=n_channels,
                                  num_repetitions=1,
                                  activation='sigmoid')

    output_shape_batched = affs_batched.get_shape().as_list()
    output_shape = output_shape_batched[1:]  # strip the batch dimension

    affs = tf.reshape(affs_batched, output_shape)

    gt_affs = tf.placeholder(tf.float32, shape=output_shape)

    if have_weights:
        loss_weights = tf.placeholder(tf.float32, shape=output_shape)
        loss = tf.losses.mean_squared_error(
            gt_affs,
            affs,
            loss_weights)
    else:
        loss = tf.losses.mean_squared_error(
            gt_affs,
            affs)
    tf.summary.scalar('loss_total', loss)

    opt = get_default_adam()
    optimizer = opt.minimize(loss)
    merged = tf.summary.merge_all()

    tf.train.export_meta_graph(filename='%s.meta' % name)

    names = {'raw': raw.name,
             'affs': affs.name,
             'gt_affs': gt_affs.name,
             'loss': loss.name,
             'optimizer': optimizer.name,
             'summary': merged.name}
    if have_weights:
        names.update({'loss_weights': loss_weights.name})

    with open('net_io_names.json', 'w') as f:
        json.dump(names, f)


def build_unet_mala_inference(name='unet_mala_inference', sample_to_isotropy=False):
    in_shape = (88, 808, 808)
    n_channels = 12

    # These values reproduce jans network
    initial_fmaps = 12
    fmap_increase = 5
    downsample_factors = [[1, 3, 3], [1, 3, 3], [3, 3, 3]] if sample_to_isotropy else \
        [[1, 3, 3], [1, 3, 3], [1, 3, 3]]

    raw = tf.placeholder(tf.float32, shape=in_shape)
    raw_batched = tf.reshape(raw, (1, 1,) + in_shape)

    unet = unet_mala(raw_batched, initial_fmaps, fmap_increase, downsample_factors)

    affs_batched = conv_pass_mala(unet,
                                  kernel_size=1,
                                  num_fmaps=n_channels,
                                  num_repetitions=1,
                                  activation='sigmoid')

    output_shape_batched = affs_batched.get_shape().as_list()
    output_shape = output_shape_batched[1:]  # strip the batch dimension

    tf.reshape(affs_batched, output_shape)
    tf.train.export_meta_graph(filename='%s.meta' % name)


def build_unet_dtu2(name='unet_dtu2', n_channels=12, have_weights=True):
    input_shape = (43, 430, 430)
    raw = tf.placeholder(tf.float32, shape=input_shape)
    raw_batched = tf.reshape(raw, (1, 1,) + input_shape)

    last_fmap, fov, anisotropy = unet_dtu2(raw_batched, 12, 6, [[1, 3, 3], [1, 3, 3], [3, 3, 3]],
                                           [[(1, 3, 3), (1, 3, 3)], [(1, 3, 3), (1, 3, 3)],
                                            [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                           [[(1, 3, 3), (1, 3, 3)], [(1, 3, 3), (1, 3, 3)],
                                            [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                           voxel_size=(10, 1, 1), fov=(10, 1, 1))

    affs_batched, fov = conv_pass_dtu2(last_fmap,
                                       kernel_size=[[1, 1, 1]],
                                       num_fmaps=n_channels,
                                       activation='sigmoid',
                                       fov=fov,
                                       voxel_size=anisotropy)

    output_shape_batched = affs_batched.get_shape().as_list()
    output_shape = output_shape_batched[1:]  # strip the batch dimension

    affs = tf.reshape(affs_batched, output_shape)
    gt_affs = tf.placeholder(tf.float32, shape=output_shape)

    if have_weights:
        loss_weights = tf.placeholder(tf.float32, shape=output_shape)
        loss = tf.losses.mean_squared_error(
            gt_affs,
            affs,
            loss_weights)
    else:
        loss = tf.losses.mean_squared_error(
            gt_affs,
            affs)
    tf.summary.scalar('loss_total', loss)

    opt = get_default_adam()
    optimizer = opt.minimize(loss)
    merged = tf.summary.merge_all()

    tf.train.export_meta_graph(filename='%s.meta' % name)

    names = {'raw': raw.name,
             'affs': affs.name,
             'gt_affs': gt_affs.name,
             'loss': loss.name,
             'optimizer': optimizer.name,
             'summary': merged.name}
    if have_weights:
        names.update({'loss_weights': loss_weights.name})

    with open('net_io_names.json', 'w') as f:
        json.dump(names, f)


def build_unet_dtu2_inference(name='unet_dtu2_inference', n_channels=12):
    input_shape = (91, 862, 862)
    raw = tf.placeholder(tf.float32, shape=input_shape)
    raw_batched = tf.reshape(raw, (1, 1,) + input_shape)

    last_fmap, fov, anisotropy = unet_dtu2(raw_batched, 12, 6, [[1, 3, 3], [1, 3, 3], [3, 3, 3]],
                                           [[(1, 3, 3), (1, 3, 3)], [(1, 3, 3), (1, 3, 3)],
                                            [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                           [[(1, 3, 3), (1, 3, 3)], [(1, 3, 3), (1, 3, 3)],
                                            [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                           voxel_size=(10, 1, 1), fov=(10, 1, 1))

    affs_batched, fov = conv_pass_dtu2(last_fmap,
                                       kernel_size=[[1, 1, 1]],
                                       num_fmaps=n_channels,
                                       activation='sigmoid',
                                       fov=fov,
                                       voxel_size=anisotropy)

    output_shape_batched = affs_batched.get_shape().as_list()

    output_shape = output_shape_batched[1:]
    tf.reshape(affs_batched, output_shape)
    tf.train.export_meta_graph(filename='%s.meta' % name)
