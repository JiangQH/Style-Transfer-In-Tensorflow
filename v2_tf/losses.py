import tensorflow as tf
from vgg import Vgg
from util import preprocess

def gram(layer):
    """
    note the num here is one
    :param layer:
    :return:
    """
    num, height, width, channels = layer.get_shape().as_list()
    filters = tf.reshape(layer, tf.stack([num, -1, channels]))
    gram_ = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(height * width * channels)
    return gram_


def get_style_feature(Config):
    """
    Config contains all the info we need
    :param Config: Config contains all the info we may need
    :return: the style feature
    """
        # get the image tensor
    with tf.Graph().as_default() as g:
    	img_bytes = tf.read_file(Config.style_image)
    	if Config.style_image.lower().endswith('png'):
		image = tf.image.decode_png(img_bytes, channels=3)
    	else:
        	image = tf.image.decode_jpeg(img_bytes, channels=3)
        # preprocess the image tensor
    	image = preprocess(image, Config)

        # init the style and get the layer_info
    	shape = [1] + image.get_shape().as_list()
    	image = tf.reshape(image, shape)
    	style_net = Vgg(Config.feature_path)
    	layer_infos = style_net.build(image)
        # get the feature we need
    	style_features = {}
    	for layer in Config.style_layers:
		feature = layer_infos[layer]
         	gram_ = tf.squeeze(gram(feature), [0])
         	style_features[layer] = gram_

        # get the feature with run
 	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True	
	with tf.Session(config=config) as sess:
    		return sess.run(style_features)


def content_loss(layer_infos, content_layers):
    """
    compute the content loss
    :param layer_infos:
    :param content_layers:
    :return:
    """
    loss = 0
    for layer in content_layers:
        generated, ori = tf.split(value=layer_infos[layer], num_or_size_splits=2, axis=0)
        size = tf.size(generated)
        loss += tf.nn.l2_loss(generated - ori) * 2 / tf.to_float(size)
    return loss

def style_loss(layer_infos, style_layers, style_features):
    """
    compute the style loss
    :param layer_infos:
    :param style_layers:
    :param style_feature:
    :return:
    """
    loss = 0
    for layer in style_layers:
        generated, _ = tf.split(value=layer_infos[layer], num_or_size_splits=2, axis=0)
        style_gram = style_features[layer]
        size = tf.size(generated)
        loss += tf.nn.l2_loss(gram(generated) - style_gram) / tf.to_float(size)
    return loss


def tv_loss(bottom):
    """
    the tv loss
    :param bottom:
    :return:
    """
    _, height, width, _ = bottom.get_shape().as_list()
    y = tf.slice(bottom, [0, 0, 0, 0], tf.stack([-1, height - 1, -1, -1])) - tf.slice(bottom, [0, 1, 0, 0],
                                                                                     [-1, -1, -1, -1])
    x = tf.slice(bottom, [0, 0, 0, 0], tf.stack([-1, -1, width - 1, -1])) - tf.slice(bottom, [0, 0, 1, 0],
                                                                                    [-1, -1, -1, -1])

    loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))

    return loss








