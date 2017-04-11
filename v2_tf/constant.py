import tensorflow as tf
import numpy as np
flags = tf.app.flags

WEIGHT_INIT_STDEV = .1

STYLE_LAYERS = ('relu1_2', 'relu2_2', 'relu3_3', 'relu4_3')
CONTENT_LAYERS = ('relu2_2')
