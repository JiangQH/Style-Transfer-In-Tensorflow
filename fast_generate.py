import tensorflow as tf
import matplotlib.pyplot as plt
from src.fast.model import inference_trainnet
import os.path as osp
import os
import numpy as np
from PIL import Image
import scipy.misc
import cv2
import argparse

IMG_WIDTH = 1024
IMG_HEIGHT = 680
MEAN = [123.68, 116.779, 103.939]
BATCH_SIZE = 2

parser = argparse.ArgumentParser(usage='python fast_generate.py -m [path/to/modeldirs] -i [path/to/inputdirs] -o [path/to/outdirs] ')
parser.add_argument('-m', '--model_dir', type=str, required=True, help='path to the model dir')
parser.add_argument('-i', '--input', type=str, required=True, help='path to the images to transfer')
parser.add_argument('-o', '--output', type=str, required=True, help='path to save the styled images')

def predict(args):
    out_dir = args.output
    img_dirs = args.input
    check_dir = args.model_dir
    with tf.Graph().as_default() as g:
        images = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 3))
        images = images - MEAN
        generated = inference_trainnet(images)
        init_op = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(init_op)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(check_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print 'restoring model params, loading ...'
            saver.restore(sess, ckpt.model_checkpoint_path)
            print 'model restored'
        else:
            raise NotImplementedError("No pretrained model, exit")
        if not osp.exists(out_dir):
            os.mkdir(out_dir)
        imgs = os.listdir(img_dirs)
        feed_data = np.zeros((BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
        for i in xrange(0, BATCH_SIZE, len(imgs)):
            im_names = []
            for j in xrange(i, BATCH_SIZE):
                if j >= len(imgs):
                    break
                img = imgs[j]
                im_names.append(img)
                im_name = osp.join(img_dirs, img)
                im = np.asarray(Image.open(im_name), dtype=np.float32)
                #w, h, c = im.shape
                im = cv2.resize(im, (IMG_WIDTH, IMG_HEIGHT))
                feed_data[j,...] = im
            gen = generated.eval(session=sess, feed_dict={images: feed_data})
            for j, img in enumerate(im_names):
                pre = gen[j, ...]
                #save = np.uint8(cv2.resize(gen, (h, w)))
                #plt.imshow(save)
                out_name = osp.join(out_dir, img)
                scipy.misc.imsave(out_name, np.uint8(pre))

        print 'done'

if __name__ == '__main__':
    args = parser.parse_args()
    predict(args)
