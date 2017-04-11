import os.path as osp
import os
import tensorflow as tf

class Dataset(object):
    def __init__(self, batch_size, mean=[123.68, 116.779, 103.939],
                 scale=1.0, width=256, height=256):
        self.batch_size = batch_size
        self.mean = mean
        self.scale = scale
        self.width = width
        self.height = height


    def input_pipelines(self, img_dirs):
        # prepare the file lists tensor and generate the filename_queue
        filenames = [osp.join(img_dirs, f) for f in os.listdir(img_dirs) if osp.isfile(f)]
        filename_queue = tf.train.string_input_producer(filenames)
        # get the data
        data = self._file_reader(filename_queue)
        # generate the batch queue
        images = tf.train.batch(data,
                                batch_size=self.batch_size)
        return images


    def _file_reader(self, filename_queue):
        # read file from queue
        reader = tf.WholeFileReader()
        _, img_bytes = reader.read(filename_queue)
        # decode it
        image_data = tf.image.decode_jpeg(img_bytes, channels=3)
        # preprocess it and return
        return self._preprocess(image_data)



    def _preprocess(self, image_tensor):
        # first resize and cast to float32
        image_tensor = tf.image.resize_images(image_tensor, (self.width, self.height))
        image_tensor = tf.cast(image_tensor, tf.float32)
        # subtract the mean and return
        assert image_tensor.get_shape().ndims == 3
        channels = image_tensor.get_shape()[-1]
        assert len(self.mean) == channels
        channel_val = tf.split(image_tensor, channels, 2)
        for i in range(channels):
            channel_val[i] -= self.mean[i]
        return tf.concat(channel_val, 2)

