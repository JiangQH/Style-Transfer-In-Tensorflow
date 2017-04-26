import tensorflow as tf
from vgg import Vgg
from util import readimg

def gram(layer):
    """
    note the num here is one
    :param layer:
    :return:
    """
    shape = tf.shape(layer)
    num = shape[0]
    num_filters = shape[3]
    height = shape[1]
    width = shape[2]
    filters = tf.reshape(layer, tf.stack([num, -1, num_filters]))
    gram_ = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(height * width * num_filters)
    return gram_


def get_style_feature(Config):
    """
    Config contains all the info we need
    :param Config: Config contains all the info we may need
    :return: the style feature
    """
    # get the image tensor
    with tf.Graph().as_default() as g:
        # get the style image
        style_img = Config.style_image
        images = tf.stack([readimg(style_img, Config)])
        # init the style and get the layer_info
        style_net = Vgg(Config.feature_path)
        layer_infos = style_net.build(images)
        # get the feature we need
    	style_features = {}
        for layer in Config.style_layers:
            layer_info = layer_infos[layer]
            style_features[layer] = gram(layer_info)
        # get the feature with run
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            return sess.run(style_features)


def content_loss(layers, content_layers):
    """
    compute the content loss
    :param layer_infos:
    :param content_layers:
    :return:
    """
    loss = 0
    for layer in content_layers:
        generated, ori = tf.split(value=layers[layer], num_or_size_splits=2, axis=0)
        size = tf.size(generated)
        loss += tf.nn.l2_loss(generated - ori) * 2 / tf.to_float(size)
    return loss

def style_loss(layers, style_layers, style_features):
    """
    compute the style loss
    :param layers:
    :param style_layers:
    :param style_feature:
    :return:
    """
    loss = 0
    for style_gram, layer in zip(style_features, style_layers):
        generated, _ = tf.split(value=layers, num_or_size_splits=2, axis=0)
        size = tf.size(generated)
        for style_im in style_gram:
            loss += tf.nn.l2_loss(tf.reduce_sum(gram(generated) - style_im, 0)) / tf.to_float(size)
    return loss


def tv_loss(bottom):
    """
    the tv loss
    :param bottom:
    :return:
    """
    shape = tf.shape(bottom)
    height = shape[1]
    width = shape[2]
    y = tf.slice(bottom, [0, 0, 0, 0], tf.stack([-1, height - 1, -1, -1])) - tf.slice(bottom, [0, 1, 0, 0],
                                                                                     [-1, -1, -1, -1])
    x = tf.slice(bottom, [0, 0, 0, 0], tf.stack([-1, -1, width - 1, -1])) - tf.slice(bottom, [0, 0, 1, 0],
                                                                                    [-1, -1, -1, -1])

    loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))

    return loss








