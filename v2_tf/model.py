import tensorflow as tf

def _get_variable(name, shape, initializer, trainable):
    var = tf.get_variable(name, shape=shape, initializer=initializer, trainable=trainable, dtype=tf.float32)
    return var

def _weights_with_weight_decay(name, shape, stddev, trainable):
    """
    :param name:
    :param shape:
    :param stddev:
    :param trainable:
    :return:
    """
    var = _get_variable(name, shape, tf.truncated_normal_initializer(stddev=stddev), trainable=trainable)
    return var


def _instance_norm(scope_name, inputs, reuse, trainable=True):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        channels = inputs.get_shape().as_list()[3]
        shift = _get_variable('shift', [channels], tf.constant_initializer(0.0), trainable=trainable)
        scale = _get_variable('scale', [channels], tf.constant_initializer(1.0), trainable=trainable)
        epsilon = 1e-3
        mean, var = tf.nn.moments(inputs, [1, 2], keep_dims=True)
        normalized = (inputs - mean) / (var + epsilon) ** (.5)
        return scale * normalized + shift


def _conv2d(scope_name, inputs, input_channels, output_channels, kernel, stride,
           reuse=False, bias_term=False, trainable=True):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        # get the shape and kernel weights
        weight_shape = [kernel, kernel, input_channels, output_channels]
        kernel = _weights_with_weight_decay('weights',
                                             shape=weight_shape,
                                             stddev=0.01,
                                             trainable=trainable)
        # get the conv out
        conv = tf.nn.conv2d(inputs, kernel, [1, stride, stride, 1], padding='SAME')
        # if we need the biases
        if bias_term:
            # get the bias term
            bias_shape = [output_channels]
            biases = _get_variable('biases', bias_shape, tf.constant_initializer(0.1), trainable=trainable)
            conv = tf.nn.bias_add(conv, biases)
        return conv


def _deconv_unit(scope_name, inputs, input_channels, output_channels, kernel, stride, reuse=False,
                 trainable=True):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        weight_shape = [kernel, kernel, output_channels, input_channels]
        kernel = _weights_with_weight_decay('weights', weight_shape, stddev=0.01, trainable=trainable)
        batches, height, width, channels = inputs.get_shape().as_list()
        out_shape = tf.stack([batches, height*stride, width*stride, output_channels])
        strides = [1, stride, stride, 1]
        conv = tf.nn.conv2d_transpose(inputs, kernel, out_shape, strides, padding='SAME')
        conv = _instance_norm('instance_norm', conv, reuse=reuse)

        return tf.nn.relu(conv)



def _conv_unit(scope_name, inputs, input_channels, output_channels, kernel, stride, relu=True,
              reuse=False, trainable=True):
    with tf.variable_scope(scope_name) as scope:
        # a first conv layer
        conv = _conv2d(scope.name, inputs, input_channels, output_channels, kernel, stride,
                      trainable=trainable)
        # norm it
        conv = _instance_norm('instance_norm', conv, reuse=reuse)
        if relu:
            conv = tf.nn.relu(conv, name=scope.name)
        return conv




def _residual_block(scope_name, inputs, input_channels, kernel=3, stride=1, reuse=False, trainable=True):
    with tf.variable_scope(scope_name) as scope:
        # a first conv unit with relu
        conv1 = _conv_unit('conv_unit1', inputs, input_channels, input_channels, kernel, stride, relu=True,
                          reuse=reuse, trainable=trainable)
        # a second conv unit without relu
        conv2 = _conv_unit('conv_unit2', conv1, input_channels, input_channels, kernel, stride, relu=False,
                           reuse=reuse, trainable=trainable)
        return conv1 + conv2


def _remap_unit(scope_name, bottom):
    with tf.variable_scope(scope_name) as scope:
        out = tf.nn.tanh(bottom, name=scope.name)
        return tf.multiply(tf.add(out, 1), 127.5)





def inference_trainnet(images):
    """
    build the fist-stage training net
    note use tf.get_variable() instead of tf.Variable() in order to share variables across multiple GPU
    training runs
    :param images: images from the imagedata_pipelines
    :return: generated images
    """
    conv1 = _conv_unit('conv1', images, 3, 32, 9, 1)
    conv2 = _conv_unit('conv2', conv1, 32, 64, 3, 2)
    conv3 = _conv_unit('conv3', conv2, 64, 128, 3, 2)
    res1 = _residual_block('res1', conv3, 128)
    res2 = _residual_block('res2', res1, 128)
    res3 = _residual_block('res3', res2, 128)
    res4 = _residual_block('res4', res3, 128)
    res5 = _residual_block('res5', res4, 128)
    deconv1 = _deconv_unit('deconv1', res5, 128, 64, 3, 2)
    deconv2 = _deconv_unit('deconv2', deconv1, 64, 32, 3, 2)
    out = _conv_unit('out', deconv2, 32, 3, 9, 1)
    generated = _remap_unit('remap', out)
    return generated













