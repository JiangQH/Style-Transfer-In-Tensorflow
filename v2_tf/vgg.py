import tensorflow as tf
import numpy as np
class Vgg(object):

    def __init__(self, model_path):
        # load the numpy weights
        self.data_dict = np.load(model_path, encoding='latin1').items()


    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name='filter')

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name='biases')

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            kernel = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, kernel, [1, 1, 1, 1], padding='SAME')
            bias = self.get_bias(name)
            conv = tf.nn.bias_add(conv, bias)
            return tf.nn.relu(conv)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

    def build(self, images):
        """
        build the net from npy file,
        return the dict containing layer infos
        :param images: the image tensor, having been preprocessed
        :return:
        """
        # convert image from RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=images)
        bgr = tf.concat(axis=3, values=[blue, green, red])

        # now do the net
        conv1_1 = self.conv_layer(bgr, 'conv1_1')
        conv1_2 = self.conv_layer(conv1_1, 'conv1_2')
        pool1 = self.max_pool(conv1_2, 'pool1')

        conv2_1 = self.conv_layer(pool1, 'conv2_1')
        conv2_2 = self.conv_layer(conv2_1, 'conv2_2')
        pool2 = self.max_pool(conv2_2, 'pool2')

        conv3_1 = self.conv_layer(pool2, 'conv3_1')
        conv3_2 = self.conv_layer(conv3_1, 'conv3_2')
        conv3_3 = self.conv_layer(conv3_2, 'conv3_3')
        pool3 = self.max_pool(conv3_3, 'pool3')

        conv4_1 = self.conv_layer(pool3, 'conv4_1')
        conv4_2 = self.conv_layer(conv4_1, 'conv4_2')
        conv4_3 = self.conv_layer(conv4_2, 'conv4_3')
        pool4 = self.max_pool(conv4_3, 'pool4')

        conv5_1 = self.conv_layer(pool4, 'conv5_1')
        conv5_2 = self.conv_layer(conv5_1, 'conv5_2')
        conv5_3 = self.conv_layer(conv5_2, 'conv5_3')

        net = {}
        net['conv1_1'] = conv1_1
        net['conv1_2'] = conv1_2
        net['conv2_1'] = conv2_1
        net['conv2_2'] = conv2_2
        net['conv3_1'] = conv3_1
        net['conv3_2'] = conv3_2
        net['conv3_3'] = conv3_3
        net['conv4_1'] = conv4_1
        net['conv4_2'] = conv4_2
        net['conv4_3'] = conv4_3
        net['conv5_1'] = conv5_1
        net['conv5_2'] = conv5_2
        net['conv5_3'] = conv5_3
        return net
