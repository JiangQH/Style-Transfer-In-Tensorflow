import tensorflow as tf
import matplotlib.pyplot as plt
from src.fast.model import inference_trainnet
import os.path as osp
import os
import numpy as np
from PIL import Image
import scipy.misc
import cv2
check_dir = './models'
out_dir = './result/fast'
IMG_WIDTH = 1024
IMG_HEIGHT = 680
MEAN = [123.68, 116.779, 103.939]
def predict(img_dirs):
    with tf.Graph().as_default() as g:
        images = tf.placeholder(tf.float32, shape=(1, IMG_HEIGHT, IMG_WIDTH, 3))
        generated = inference_trainnet(images)
        init_op = tf.global_variables_initializer()
        sess = tf.Session()
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
        for img in imgs:
            im_name = osp.join(img_dirs, img)
            im = np.asarray(Image.open(im_name), dtype=np.float32)
            #w, h, c = im.shape
            im = cv2.resize(im, (IMG_WIDTH, IMG_HEIGHT))
            data = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
            data[0, :, :, 0] = im[..., 0] - MEAN[0]
            data[0, :, :, 1] = im[..., 1] - MEAN[1]
            data[0, :, :, 2] = im[..., 2] - MEAN[2]
            gen = generated.eval(session=sess, feed_dict={images: data})
            gen = gen[0, ...]
            #save = np.uint8(cv2.resize(gen, (h, w)))
            #plt.imshow(save)
            out_name = osp.join(out_dir, img)
            scipy.misc.imsave(out_name, np.uint8(gen))

        print 'done'

predict('./eval')
