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
    assert num == 1
    filters = tf.reshape(layer, tf.stack([num, -1, channels]))
    gram = tf.matmul(filters, filters, transpose_a=True) / tf.cast(height * width * channels, tf.float32)
    return gram


def get_style_feature(Config):
    """
    Config contains all the info we need
    :param Config: Config contains all the info we may need
    :return: the style feature
    """
    with tf.Graph().as_default():
        # get the image tensor
        img_bytes = tf.read_file(Config.style_image)
        if Config.style_image.lower().endswith('png'):
            image = tf.image.decode_png(img_bytes)
        else:
            image = tf.image.decode_jpeg(img_bytes)
        # preprocess the image tensor
        image = preprocess(image, Config)

        # init the style and get the layer_info
        style_net = Vgg(Config.model_path)
        layer_infos = style_net.build(image)
        # get the feature we need
        style_features = []
        for layer in Config.style_layers:
            feature = layer_infos[layer]
            feature = tf.squeeze(gram(feature), [0])
            style_features.append(feature)

        # get the feature with run
        sess = tf.Session()
        return sess.run(style_features)












