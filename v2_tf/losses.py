import tensorflow as tf
from util import preprocess
from loss_net import vgg16
from constant import STYLE_LAYERS

def _gram(layer):
    """
    note the num here is one
    :param layer:
    :return:
    """
    num, height, width, channels = layer.get_shape().as_list()
    assert num == 1
    filters = tf.reshape(layer, tf.stack([-1, channels]))
    gram = tf.matmul(filters, filters, transpose_a=True) / tf.cast(height * width * channels, tf.float32)
    return gram


def get_style_features(img_src, net_src):
    """
    get the style features for the style image
    :param FLAGS:
    :return:
    """
    with tf.Graph().as_default():
        style_img = tf.read_file(img_src)
        image = tf.image.decode_png(style_img)
        nets = vgg16(image, net_src)
        features = []
        for layer in STYLE_LAYERS:
            feature = nets[layer]
            features.append(_gram(feature))

        # now do the computing job
        with tf.Session() as sess:



