from __future__ import print_function
import tensorflow as tf

from .unet_dtu2 import conv_pass, downsample, upsample, crop_zyx


def unet_multiscale(fmaps_in,
                    num_fmaps,
                    fmap_inc_factor,
                    downsample_factors,
                    kernel_size_down,
                    kernel_size_up,
                    activation='relu',
                    layer=0,
                    fov=(1, 1, 1),
                    voxel_size=(1, 1, 1),
                    outputs=None):

    '''Create a U-Net::
        f_in --> f_left --------------------------->> f_right--> f_out
                    |                                   ^
                    v                                   |
                 g_in --> g_left ------->> g_right --> g_out
                             |               ^
                             v               |
                                   ...
    where each ``-->`` is a convolution pass (see ``conv_pass``), each `-->>` a
    crop, and down and up arrows are max-pooling and transposed convolutions,
    respectively.
    The U-Net expects tensors to have shape ``(batch=1, channels, depth, height,
    width)``.
    This U-Net performs only "valid" convolutions, i.e., sizes of the feature
    maps decrease after each convolution.
    Args:
        fmaps_in:
            The input tensor.
        num_fmaps:
            The number of feature maps in the first layer. This is also the
            number of output feature maps.
        fmap_inc_factor:
            By how much to multiply the number of feature maps between layers.
            If layer 0 has ``k`` feature maps, layer ``l`` will have
            ``k*fmap_inc_factor**l``.
        downsample_factors:
            List of lists ``[z, y, x]`` to use to down- and up-sample the
            feature maps between layers.
        kernel_size_down:
            List of lists of tuples ``(z, y, x)`` of kernel sizes. The number of
            tuples in a list determines the number of convolutional layers in the
            corresponding level of the unet on the left side.
        kernel_size_up:
            List of lists of tuples ``(z, y, x)`` of kernel sizes. The number of
            tuples in a list determines the number of convolutional layers in the
            corresponding level of the unet on the right side. Within one of the
            lists going from left to right.
        activation:
            Which activation to use after a convolution. Accepts the name of any
            tensorflow activation function (e.g., ``relu`` for ``tf.nn.relu``).
        layer:
            Used internally to build the U-Net recursively.
        fov:
            Initial field of view in physical units
        voxel_size:
            Size of a voxel in the input data, in physical units
    '''

    prefix = "    "*layer
    print(prefix + "Creating U-Net layer %i" % layer)
    print(prefix + "f_in: " + str(fmaps_in.shape))

    if layer == 0:
        assert outputs is None
        outputs = []

    # convolve
    with tf.name_scope("lev%i" % layer):

        f_left, fov = conv_pass(
            fmaps_in,
            kernel_size=kernel_size_down[layer],
            num_fmaps=num_fmaps,
            activation=activation,
            name='unet_layer_%i_left' % layer,
            fov=fov,
            voxel_size=voxel_size,
            prefix=prefix)

        # last layer does not recurse
        bottom_layer = (layer == len(downsample_factors))

        if bottom_layer:
            print(prefix + "bottom layer")
            print(prefix + "f_out: " + str(f_left.shape))
            outputs.append(f_left)
            return f_left, fov, voxel_size

        # downsample

        g_in, fov, voxel_size = downsample(
            f_left,
            downsample_factors[layer],
            'unet_down_%i_to_%i' % (layer, layer + 1),
            fov=fov,
            voxel_size=voxel_size,
            prefix=prefix)

        # recursive U-net
        g_out, fov, voxel_size = unet_multiscale(
            g_in,
            num_fmaps=num_fmaps*fmap_inc_factor,
            fmap_inc_factor=fmap_inc_factor,
            downsample_factors=downsample_factors,
            kernel_size_down=kernel_size_down,
            kernel_size_up=kernel_size_up,
            activation=activation,
            layer=layer+1,
            outputs=outputs,
            fov=fov,
            voxel_size=voxel_size)

        print(prefix + "g_out: " + str(g_out.shape))

        # upsample
        g_out_upsampled, voxel_size = upsample(
            g_out,
            downsample_factors[layer],
            num_fmaps,
            activation=activation,
            name='unet_up_%i_to_%i' % (layer + 1, layer),
            fov=fov,
            voxel_size=voxel_size,
            prefix=prefix)

        print(prefix + "g_out_upsampled: " + str(g_out_upsampled.shape))

        # copy-crop
        f_left_cropped = crop_zyx(f_left, g_out_upsampled.get_shape().as_list())

        print(prefix + "f_left_cropped: " + str(f_left_cropped.shape))

        # concatenate along channel dimension
        f_right = tf.concat([f_left_cropped, g_out_upsampled], 1)

        print(prefix + "f_right: " + str(f_right.shape))

        # convolve
        f_out,  fov = conv_pass(
            f_right,
            kernel_size=kernel_size_up[layer],
            num_fmaps=num_fmaps,
            name='unet_layer_%i_right' % layer,
            fov=fov,
            voxel_size=voxel_size,
            prefix=prefix
            )
        print(prefix + "f_out: " + str(f_out.shape))
        outputs.append(f_out)

    if layer == 0:
        # we invert the outputs, because we want the highest resolution
        # to be outpurs[0]
        return outputs[::-1], fov, voxel_size
    else:
        return f_out, fov, voxel_size


# TODO what is the proper way to combine the losses in tf?
# there is tf.reduce_sum, but according to `the internet` `+`
# should also work
def multiscale_loss_weighted(gt_affs, outputs, loss_weights):
    assert len(gt_affs) == len(outputs) == len(loss_weights) == 4
    loss = tf.losses.mean_squared_error(gt_affs[0], outputs[0], loss_weights[0]) +\
        tf.losses.mean_squared_error(gt_affs[1], outputs[1], loss_weights[1]) +\
        tf.losses.mean_squared_error(gt_affs[2], outputs[2], loss_weights[2]) +\
        tf.losses.mean_squared_error(gt_affs[3], outputs[3], loss_weights[3])
    return loss


# TODO what is the proper way to combine the losses in tf?
# there is tf.reduce_sum, but according to `the internet` `+`
# should also work
def multiscale_loss(gt_affs, outputs):
    assert len(gt_affs) == len(outputs) == 4
    loss = tf.losses.mean_squared_error(gt_affs[0], outputs[0]) +\
        tf.losses.mean_squared_error(gt_affs[1], outputs[1]) +\
        tf.losses.mean_squared_error(gt_affs[2], outputs[2]) +\
        tf.losses.mean_squared_error(gt_affs[3], outputs[3])
    return loss
