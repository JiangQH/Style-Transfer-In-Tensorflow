import tensorflow as tf
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






