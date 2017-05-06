import argparse
import random
import os.path as osp
import os
import cv2
import time
import shutil
from fast_style import predict

parser = argparse.ArgumentParser(usage='python fast_style_video.py -m [path/to/modeldirs] -i [path/to/input videos] -o [path/to/outdirs] ')
parser.add_argument('-i', '--input', type=str, required=True, help='path to the video')
parser.add_argument('-o', '--output', type=str, required=True, help='path to output the video')
parser.add_argument('-m', '--model_dir', type=str, required=True, help='path to the model dir')
parser.add_argument('-b', '--batch_size', type=int, required=False, default=4, help='batch size to generate')
parser.add_argument('-d', '--device', type=str, required=False, default='/gpu:0', help='the device to run model')

TMP_DIR = '.tmp_frams_%s/' %random.randint(0, 89898)


def main():
	args = parser.parse_args()
	device = args.device
	batch_size = args.batch_size
	check_dir = args.model_dir
	video_path = args.input
	video_out = args.output

	# make tmp dirs to do the job
	if not osp.exists(TMP_DIR):
		os.mkdir(TMP_DIR)
	in_dir = osp.join(TMP_DIR, 'in')
	if not osp.exists(in_dir):
		os.mkdir(in_dir)
	out_dir = osp.join(TMP_DIR, 'out')
	if not osp.exists(out_dir):
		os.mkdir(out_dir)
	
	# release the frame file to in dir
	print 'extracting video frames...'
	start_time = time.time()
	vidcap = cv2.VideoCapture(video_path)
	fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
	width = vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
	height = vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
	print '{} {} {}'.format(fps, width, height)
	success = True
	count = 0
	base_names = []
	while True:
		success, image = vidcap.read()
		if not success:
			break
		# print 'read a new fram: ', success
		# save to the in dir
		cv2.imwrite(osp.join(in_dir, 'frames%d.png'%count), image)
		base_names.append('frames%d.png'%count)
		count += 1
	print 'extracting video done, time consumes {} s'.format(time.time() - start_time)

	in_imgs = [osp.join(in_dir, name) for name in base_names]
	out_imgs = [osp.join(out_dir, name) for name in base_names]
	print 'doing the transfer job...'
	start_time = time.time()
	predict(in_imgs, out_imgs, check_dir, batch_size, device)
	print 'transfer done, time consumes {}'.format(time.time() - start_time)

	# rebuild a video
	video = cv2.VideoWriter(osp.join(video_out, 'styled.avi'), cv2.cv.CV_FOURCC('M','J','P','G'), fps, (int(width), int(height)))
	for img in out_imgs:
		im = cv2.imread(img)
		video.write(im)
	cv2.destroyAllWindows()
	video.release()
	shutil.rmtree(TMP_DIR)
	print 'done'

if __name__ == '__main__':
	main()
	
	


