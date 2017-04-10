import os.path as osp
import os
import tensorflow as tf
from util import preprocess

def batch_images(batch_size, new_height, new_width, src):
    files = [osp.join(src, f) for f in os.listdir(src) if osp.isfile(osp.join(src, f))]
    file_names = tf.train.string_input_producer(files, shuffle=True)
    img_file = tf.read_file(file_names)
    image = tf.image.decode_png(img_file, channels=3)
    image = tf.image.resize_images(image, (new_height, new_width))
    image = tf.cast(image, tf.float32)
    image = preprocess(image)
    return tf.train.batch([image], batch_size=batch_size)