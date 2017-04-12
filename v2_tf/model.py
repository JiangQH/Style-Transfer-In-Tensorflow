import tensorflow as tf

def _get_variable(name, shape, initializer, trainable):
    var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
    return var

def _weights_with_weight_decay(name, shape, stddev, wd, trainable):
    """
    get the weight decay here, so we can use the tf auto summary the weight decay for us
    :param name:
    :param shape:
    :param stddev:
    :param wd:
    :param trainable:
    :return:
    """
    var = _get_variable(name, shape, tf.truncated_normal(stddev=stddev), trainable=trainable)
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _get_weights_shape(bottom, num_output, kernel_size, transpose=False):
    """
    get the shape of weights of a conv or deconv layer
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

    return weight_shape

def _instance_norm(scope_name, bottom, reuse, trainable=True):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        _, _, _, channels = bottom.get_shape().as_list()
        shift = _get_variable('shift', [channels], tf.constant_initializer(0.0), trainable=trainable)
        scale = _get_variable('scale', [channels], tf.constant_initializer(0.0), trainable=trainable)
        epsilon = 1e-6
        mean, var = tf.nn.moments(bottom, [1, 2], keep_dims=True)
        normalized = tf.divide(tf.subtract(bottom, mean), tf.sqrt(tf.add(var, epsilon)))
        return tf.add(tf.multiply(normalized, scale), shift)


def _conv2d(scope_name, bottom, kernel_size, num_output, stride, relu=True,
           wd=0.0, reuse=False, bias_term=False, trainable=True):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        # get the shape and kernel weights
        weight_shape = _get_weights_shape(bottom, num_output, kernel_size)
        kernel = _weights_with_weight_decay('weights',
                                             shape=weight_shape,
                                             stddev=0.01,
                                             wd=wd, trainable=trainable)
        # get the conv out
        conv = tf.nn.conv2d(bottom, kernel, stride, padding='SAME')
        # if we need the biases
        if bias_term:
            # get the bias term
            bias_shape = [num_output]
            biases = _get_variable('biases', bias_shape, tf.constant_initializer(0.1), trainable=trainable)
            conv = tf.nn.bias_add(conv, biases)

        # if we need the relu layer
        if relu:
            conv = tf.nn.relu(conv, name=scope.name)
        return conv

def _max_pool(scope_name, bottom, kernel_size, stride):
    with tf.variable_scope(scope_name):
        pool = tf.nn.max_pool(bottom, [1, kernel_size, kernel_size, 1],
                              [1, stride, stride, 1], padding='SAME')
        return pool

def _deconv_unit(scope_name, bottom, kernel_size, num_output, stride, wd=0.0, reuse=False,
                 trainable=True):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        weight_shape = _get_weights_shape(bottom, num_output, kernel_size, transpose=True)
        kernel = _weights_with_weight_decay('weights', weight_shape, stddev=0.01, wd=wd, trainable=trainable)

        nums, height, width, channels = bottom.get_shape().as_list()
        new_height, new_width = int(height * stride), int(width * stride)
        new_shape = tf.stack([nums, new_height, new_width, channels])

        strides = [1, stride, stride, 1]

        conv = tf.nn.conv2d_transpose(bottom, kernel, new_shape, strides)
        conv = _instance_norm('instance_norm', conv, reuse=reuse)

        return conv



def _conv_unit(scope_name, bottom, kernle_size, num_ouput, stride, relu=True,
              reuse=False, trainable=True):
    with tf.variable_scope(scope_name) as scope:
        # a first conv layer
        conv = _conv2d(scope.name, bottom, kernle_size, num_ouput, stride, reuse=reuse,
                      trainable=trainable)
        # norm it
        conv = _instance_norm('instance_norm', conv, reuse=reuse)
        if relu:
            conv = tf.nn.relu(conv, name=scope.name)
        return conv




def _residual_block(scope_name, bottom, kernel_size=3, num_ouput=128, stride=1, reuse=False, trainable=True):
    with tf.variable_scope(scope_name) as scope:
        # a first conv unit with relu
        conv1 = _conv_unit('conv_unit1', bottom, kernel_size, num_ouput, stride, relu=True,
                          reuse=reuse, trainable=trainable)
        # a second conv unit without relu
        conv2 = _conv_unit('conv_unit2', conv1, kernel_size, num_ouput, stride, relu=False, reuse=reuse, trainable=trainable)
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
    conv1 = _conv_unit('conv1', bottom=images, kernle_size=9, num_ouput=32, stride=1)
    conv2 = _conv_unit('conv2', bottom=conv1, kernle_size=3, num_ouput=64, stride=2)
    conv3 = _conv_unit('conv3', bottom=conv2, kernle_size=3, num_ouput=128, stride=2)
    res1 = _residual_block('res1', bottom=conv3)
    res2 = _residual_block('res2', bottom=res1)
    res3 = _residual_block('res3', bottom=res2)
    res4 = _residual_block('res4', bottom=res3)
    res5 = _residual_block('res5', bottom=res4)
    deconv1 = _deconv_unit('deconv1', bottom=res5, kernel_size=3, num_output=64, stride=2)
    deconv2 = _deconv_unit('deconv2', bottom=deconv1, kernel_size=3, num_output=32, stride=2)
    out = _conv_unit('out', bottom=deconv2, kernle_size=9, num_ouput=3, stride=1, relu=False)
    generated = _remap_unit('remap', out)
    return generated













