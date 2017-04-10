from constant import IMAGE_MEAN, IMAGE_SCALE, WEIGHT_INIT_STDEV
import tensorflow as tf

def preprocess(image):
    """
    the preprocessing step
    :param image:
    :return:
    """
    num_channels = image.get_shape().as_list[-1]
    channels = tf.split(image, num_channels, 2)
    for i in range(num_channels):
        channels[i] -= IMAGE_MEAN[i]
    return tf.multiply(tf.concat(channels, 2), IMAGE_SCALE)

def unprocess(image):
    image = tf.divide(image, IMAGE_SCALE)
    num_channels = image.get_shape().as_list[-1]
    channels = tf.split(image, num_channels, 2)
    for i in range(num_channels):
        channels[i] += IMAGE_MEAN[i]
    return tf.concat(channels, 2)


def maxpool_layer(bottom, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1)):
    """
    a wrapper to the max pooling layer
    :param bottom:
    :param ksize:
    :param strides:
    :return:
    """
    return tf.nn.max_pool(bottom, ksize=ksize, strides=strides, padding='SAME')

def conv_layer(bottom, weights, strides=(1, 1, 1, 1), bias=None):
    """
    a wrapper to the conv layer
    :param bottom:
    :param weights:
    :param strides:
    :param bias:
    :return:
    """
    conv = tf.nn.conv2d(bottom, tf.constant(weights), strides=strides, padding='SAME')
    if bias is not None:
        conv = tf.nn.bias_add(conv, bias)
    return conv

def residual_block(bottom, num_output=128, kernel_size=3, stride=1, name='residual'):
    """
    the residual block used in the net
    :param bottom:
    :param num_output:
    :param kernel_size:
    :param stride:
    :return:
    """
    with tf.variable_scope(name):
        # first conv unit with relu
        conv1 = conv_unit(bottom, num_output, kernel_size, stride)
        # second conv unit without relu
        conv2 = conv_unit(conv1, num_output, kernel_size, stride, relu=False)
        return bottom + conv2


def conv_unit(bottom, num_output, kernel_size, stride, relu=True, name='conv_unit'):
    """
    the conv unit in residual net
    :param bottom:
    :param num_output:
    :param kernel_size:
    :param stride:
    :param relu:
    :return:
    """
    with tf.variable_scope(name):
        weights = _conv_init_weights(bottom, num_output, kernel_size)
        strides = [1, stride, stride, 1]
        # a conv layer
        conv = conv_layer(bottom, weights, strides, bias=None)
        # a norm layer
        normed = _instance_norm(conv)
        if relu:
            normed = tf.nn.relu(normed)
        return normed


def deconv_unit(bottom, num_ouput, kernel_size, stride, name='deconv_unit'):
    """
    the deconv layer
    :param bottom:
    :param num_ouput:
    :param kernel_size:
    :param stride:
    :return:
    """
    with tf.variable_scope(name):
        weights = _conv_init_weights(bottom, num_ouput, kernel_size, transpose=True)
        nums, heights, widths, channels = bottom.get_shape().as_list()
        new_heights, new_widths = int(heights * stride), int(widths * stride)
        new_shape = [nums, new_heights, new_widths, channels]

        out_shape = tf.stack(new_shape)
        strides = [1, stride, stride, 1]
        conv = tf.nn.conv2d_transpose(bottom, weights, out_shape, strides, padding='SAME')
        normed = _instance_norm(conv)
        return tf.nn.relu(normed)


def _conv_init_weights(bottom, num_output, kernel_size, transpose=False):
    """
    init the weight of conv kernel
    :param bottom:
    :param num_output:
    :param kernel_size:
    :param transpose:
    :return:
    """
    num, heights, widths, channels = bottom.get_shape().as_list()
    if not transpose:
        weight_shape = [kernel_size, kernel_size, channels, num_output]
    else:
        weight_shape = [kernel_size, kernel_size, num_output, channels]
    weights = tf.Variable(tf.truncated_normal(weight_shape, stddev=WEIGHT_INIT_STDEV, seed=1), dtype=tf.float32)
    return weights


def _instance_norm(bottom, train=True):
    """
    instance norm
    :param bottom:
    :param train:
    :return:
    """
    num, heights, widths, channels = bottom.get_shape().as_list()
    mu, sigma = tf.nn.moments(bottom, [1, 2], keep_dims=True)
    shift = tf.Variable(tf.zeros([channels]))
    scale = tf.Variable(tf.zeros([channels]))
    epsilon = 1e-6
    normalized = tf.divide(tf.subtract(bottom, mu), tf.sqrt(tf.add(sigma, epsilon)))
    return tf.add(tf.multiply(normalized, scale), shift)