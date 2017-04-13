import os.path as osp
import os
import tensorflow as tf
from util import preprocess

class Dataset(object):
    def __init__(self, Config):
        self.batch_size = Config.batch_size
        self.config = Config

    def imagedata_pipelines(self):
        # prepare the file lists tensor and generate the filename_queue
        img_dirs = self.config.data_dir
        filenames = [osp.join(img_dirs, f) for f in os.listdir(img_dirs) if osp.isfile(osp.join(img_dirs, f))]
        filename_queue = tf.train.string_input_producer(filenames)
        # get the data
        data = self._file_reader(filename_queue)
        # generate the batch queue
        images = tf.train.batch([data],
                                batch_size=self.batch_size)
        return images


    def _file_reader(self, filename_queue):
        # read file from queue
        reader = tf.WholeFileReader()
        _, img_bytes = reader.read(filename_queue)
        # decode it
        image_data = tf.image.decode_jpeg(img_bytes, channels=3)
        # preprocess it and return
        return preprocess(image_data, self.config)



