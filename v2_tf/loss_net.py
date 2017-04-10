import tensorflow as tf
import numpy as np
import scipy.io
from util import conv_layer, maxpool_layer

def vgg16(inputs, model_file):
    layers = ('conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

              'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

              'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3',

              'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',

              'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3'
              )
    info = scipy.io.loadmat(model_file)
    weights = info['layers'][0]

    net = {}
    current = inputs
    with tf.variable_scope('vgg_16'):
        for i, name in enumerate(layers):
            with tf.variable_scope(name):
                kind = name[:4]
                if kind == 'conv':
                    kernels, bias = weights[i][0][0][0][0]
                    kernels = np.transpose(kernels, (1, 0, 2, 3))
                    bias = np.reshape(bias, -1)
                    current = conv_layer(current, weights=kernels, bias=bias)
                elif kind == 'relu':
                    current = tf.nn.relu(current)
                elif kind == 'pool':
                    current = maxpool_layer(current)
                net[name] = current
    return net

