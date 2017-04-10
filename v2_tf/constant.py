import tensorflow as tf
import numpy as np
flags = tf.app.flags

WEIGHT_INIT_STDEV = .1
IMAGE_MEAN = [123.68, 116.779, 103.939]
IMAGE_SCALE = 1.0

BATCH_SIZE = 16
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

STYLE_LAYERS = ('relu1_2', 'relu2_2', 'relu3_3', 'relu4_3')
CONTENT_LAYERS = ('relu2_2')
