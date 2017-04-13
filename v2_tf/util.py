import tensorflow as tf
import yaml

def preprocess(image_tensor, Config):
    # first resize and cast to float32
    image_tensor = tf.image.resize_images(image_tensor, (Config.width, Config.height))
    image_tensor = tf.cast(image_tensor, tf.float32)
    # subtract the mean and return
    assert image_tensor.get_shape().ndims == 3
    channels = image_tensor.get_shape()[-1]
    assert len(Config.mean) == channels
    channel_val = tf.split(image_tensor, channels, 2)
    for i in range(channels):
        channel_val[i] -= Config.mean[i]
    return tf.concat(channel_val, 2)



class Config(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)

def load_config(conf_file):
    with open(conf_file) as f:
        config = Config(**yaml.load(f))
    return config