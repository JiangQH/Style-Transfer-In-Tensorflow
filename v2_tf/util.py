# coding: utf-8
import tensorflow as tf
import yaml

def preprocess(image_tensor, Config):
    # first resize and cast to float32
    image_tensor = tf.image.resize_images(image_tensor, (Config.image_size, Config.image_size))
    image_tensor = tf.cast(image_tensor, tf.float32)
    # subtract the mean and return
    ndims = image_tensor.get_shape().ndims
    channels = image_tensor.get_shape()[-1].value
    assert len(Config.mean) == channels
    channel_val = tf.split(image_tensor, num_or_size_splits=channels, axis=ndims-1)
    for i in range(channels):
        channel_val[i] -= Config.mean[i]
    return tf.concat(channel_val, ndims-1)



class Config(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)

def load_config(conf_file):
    with open(conf_file) as f:
        config = Config(**yaml.load(f))
    return config