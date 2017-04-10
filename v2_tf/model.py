import tensorflow as tf
from util import conv_unit, residual_block, deconv_unit


def train_net(image):
    """
    the main net structure
    :param image:
    :return:
    """
    conv1 = conv_unit(image, num_output=32, kernel_size=9, stride=1, name='conv1')
    conv2 = conv_unit(conv1, num_output=64, kernel_size=3, stride=2, name='conv2')
    conv3 = conv_unit(conv2, num_output=128, kernel_size=3, stride=2, name='conv3')
    res1 = residual_block(conv3, num_output=128, kernel_size=3, stride=1, name='res1')
    res2 = residual_block(res1, num_output=128, kernel_size=3, stride=1, name='res2')
    res3 = residual_block(res2, num_output=128, kernel_size=3, stride=1, name='res3')
    res4 = residual_block(res3, num_output=128, kernel_size=3, stride=1, name='res4')
    res5 = residual_block(res4, num_output=128, kernel_size=3, stride=1, name='res5')
    deconv1 = deconv_unit(res5, num_ouput=64, kernel_size=3, stride=2, name='deconv1')
    deconv2 = deconv_unit(deconv1, num_ouput=32, kernel_size=3, stride=2, name='deconv2')
    conv_out = conv_unit(deconv2, num_output=3, kernel_size=9, stride=1, relu=False, name='conv_out')
    out = (tf.nn.tanh(conv_out) + 1) * 127.5
    return out




