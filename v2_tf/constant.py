import tensorflow as tf
import numpy as np
flags = tf.app.flags

WEIGHT_INIT_STDEV = .1
IMAGE_MEAN = [123.68, 116.779, 103.939]
IMAGE_SCALE = 1.0
