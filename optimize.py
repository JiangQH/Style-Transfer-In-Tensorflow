from neural_style.optimize import StyleTransfer
import argparse
import os.path as osp
import scipy.misc

parser = argparse.ArgumentParser(usage='python main.py -c [pathto/content_img] -s [pathto/style_img]')
parser.add_argument('-c', '--content_img', type=str, required=True, help='the content img to transfer')
parser.add_argument('-s', '--style_img', type=str, required=True, help='the style img to transfer from')
parser.add_argument('-m', '--model_name', type=str, required=False, default='vgg16', help='the model to use')
parser.add_argument('-gpu', '--gpu_id', type=int, required=False, default=0, help='which gpu to use')
parser.add_argument('-iter', '--max_iter', type=int, required=False, default=300, help='max iterations to run')
parser.add_argument('-v', '--verbose', type=bool, required=False, default=True, help='whether to display')
parser.add_argument('-init','-init', type=str, required=False, default='content', help='the first init img')

def main(args):
	transfer = StyleTransfer(args.content_img, args.style_img,
							 args.model_name, args.gpu_id, args.init, args.max_iter,
							 args.verbose)
	transfer.transfer()


if __name__ == "__main__":
	args = parser.parse_args()
	main(args)
