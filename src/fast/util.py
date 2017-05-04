# coding: utf-8
import tensorflow as tf
import yaml

def preprocess(image_tensor, Config):
    # first resize and cast to float32
    image_tensor = tf.image.resize_images(image_tensor, (Config.image_size, Config.image_size))
    image_tensor = tf.cast(image_tensor, tf.float32)
    # subtract the mean and return
    channels = image_tensor.get_shape()[-1].value
    assert len(Config.mean) == channels
    subtracted = image_tensor - Config.mean
    return subtracted

def readimg(path, Config):
    img_bytes = tf.read_file(path)
    image = tf.image.decode_png(img_bytes, channels=3) if path.lower().endswith('png') else \
        tf.image.decode_jpeg(img_bytes, channels=3)
    return preprocess(image, Config)

class Config(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)

def load_config(conf_file):
    with open(conf_file) as f:
        config = Config(**yaml.load(f))
    return config