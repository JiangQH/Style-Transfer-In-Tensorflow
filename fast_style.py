import tensorflow as tf
from src.fast.model import inference_trainnet
import os.path as osp
import os
import numpy as np
import scipy.misc
import argparse


MEAN = [123.68, 116.779, 103.939]
parser = argparse.ArgumentParser(usage='python fast_style.py -m [path/to/modeldirs] -i [path/to/inputdirs] -o [path/to/outdirs] ')
parser.add_argument('-m', '--model_dir', type=str, required=True, help='path to the model dir')
parser.add_argument('-i', '--input', type=str, required=True, help='path to the image to transfer')
parser.add_argument('-o', '--output', type=str, required=True, help='path to save the styled images')
parser.add_argument('-b', '--batch_size', type=int, required=False, default=4, help='batch size to generate')
parser.add_argument('-d', '--device', type=str, required=False, default='/gpu:0', help='the device to run model')


def predict(in_imgs, out_imgs, check_dir, batch_size, device):
	# get an image to figure out its size, assume that all images are of equal size
	# or it will do the resize job
	with tf.device(device), tf.Graph().as_default() as g: 
		assert len(in_imgs) == len(out_imgs)
		im_shape = scipy.misc.imread(in_imgs[0], mode='RGB').shape	
		
		feed_shape = (batch_size,) + im_shape
        images = tf.placeholder(tf.float32, shape=feed_shape)
        images = images - MEAN
        generated = inference_trainnet(images)
     
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # restore the pretrained model
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(check_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print 'restoring model params, loading ...'
            saver.restore(sess, ckpt.model_checkpoint_path)
            print 'model restored'
        else:
            raise Exception("No pretrained model...")


        # now the acture feed job
        feed_data = np.zeros(feed_shape, dtype=np.float32)
        for i in xrange(0, len(in_imgs), batch_size):
        	# fullfill a batch
            out_names = []
            im_shapes = []
            print 'handing batch {}...'.format(i / batch_size)
            for j in xrange(i, i+batch_size):
                if j >= len(in_imgs):
                    break
                in_img = in_imgs[j]
                out_img = out_imgs[j]
                out_names.append(out_img)
                im = scipy.misc.imread(in_img)
                # should we do the resize job?
                im_shapes.append(im.shape)
                if im.shape != im_shape:
                	print 'img shape not same with the original, do resizeing now, it is not recommended'
                	im = scipy.misc.imresize(im, im_shape)
                feed_data[j-i, ...] = np.asarray(im, dtype=np.float32)
            # do the eval job
            gen = generated.eval(session=sess, feed_dict={images: feed_data})
            # output the eval result
            for j, out_name in enumerate(out_names):
                pre = np.uint8(gen[j, ...])
                shape = im_shapes[j]
                # should we resize it back?
                if shape != im_shape:
                	pre = scipy.misc.imresize(pre, shape)
                scipy.misc.imsave(out_name, pre)

def main():
	args = parser.parse_args()
	device = args.device
	batch_size = args.batch_size
	check_dir = args.model_dir

	input_dir = args.input
	out_dir = args.output
	if osp.isdir(input_dir):
		base_names = os.listdir(input_dir)
		assert len(base_names) > 0, 'no imgs in the input dir'
		in_imgs = [osp.join(input_dir, name) for name in base_names]
		out_imgs = [osp.join(out_dir, name) for name in base_names]
		predict(in_imgs, out_imgs, check_dir, batch_size, device)
	else:
		in_imgs = input_dir
		path, base_name = osp.split(input_dir)
		out_imgs = osp.join(out_dir, in_imgs)
		predict(in_imgs, out_imgs, check_dir, batch_size, device)



if __name__ == '__main__':
	
	main()
