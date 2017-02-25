import caffe
import os.path as osp
import numpy as np
import scipy.misc
from numeric import Numeric
model_dir = '../model'
VGG16_CONTENTS = {"conv4_2": 1}
VGG16_STYLES = {"conv1_1": 0.2,
                "conv2_1": 0.2,
                "conv3_1": 0.2,
                "conv4_1": 0.2,
                "conv5_1": 0.2}
VGG16_LAYERS=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2', 'conv5_1']

class SimpleTools(object):
    def __init__(self, model_name, content_img, style_img, device_id, init='mixed',
                 ratio=1e-3):
        self._loadModel(model_name, device_id)
        self._prepareFGAndInitInput(content_img, style_img, init)
        self.ratio = ratio

    def _loadModel(self, model_name, id):
        print 'loading model...{}'.format(model_name)
        model_file = osp.join(model_dir, model_name+'.prototxt')
        model_weights = osp.join(model_dir, model_name+'.caffemodel')
        mean_file = osp.join(model_dir, model_name+'.npy')
        if id == -1:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(id)
        net = caffe.Net(model_file, model_weights, caffe.TEST)
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
        transformer.set_channel_swap('data', (2, 1, 0))
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_raw_scale('data', 255)
        self.net = net
        self.transformer = transformer
        if model_name == 'vgg16':
            self.style_layers = VGG16_STYLES
            self.content_layers = VGG16_CONTENTS
            self.layers = VGG16_LAYERS
        print 'model loading done'


    def _prepareFGAndInitInput(self, contentImgname, styleImgname, init='mixed'):
        print 'prepare img_content...'
        img_content = caffe.io.load_image(contentImgname)
        self.ori_width = img_content.shape[0]
        self.ori_height = img_content.shape[1]
        self.img_content = img_content
        # resize it to square
        width = self.net.blobs['data'].shape[2]
        img_content = scipy.misc.imresize(img_content, (width, width))
        # transform it
        x = self.transformer.preprocess('data', img_content)
        self.F_content = Numeric().computeFandG(x, self.net, [], self.content_layers)[0]
        print 'prepare style_content...'
        img_style = caffe.io.load_image(styleImgname)
        self.img_style = img_style
        img_style = scipy.misc.imresize(img_style, (width, width))
        # transform it
        x = self.transformer.preprocess('data', img_style)
        self.G_style = Numeric().computeFandG(x, self.net, self.style_layers, [])[1]
        print 'prepare init input...'
        if init == 'mixed':
            img0 = 0.95 * self.transformer.preprocess('data', img_content) +\
                        0.05 * self.transformer.preprocess('data', img_style)
        else:
            img0 = self.transformer.preprocess('data', img_content)
        # compute data bounds
        data_min = -self.transformer.mean["data"][:, 0, 0]
        data_max = data_min + self.transformer.raw_scale["data"]
        data_bounds = [(data_min[0], data_max[0])] * (img0.size / 3) + \
                      [(data_min[1], data_max[1])] * (img0.size / 3) + \
                      [(data_min[2], data_max[2])] * (img0.size / 3)
        self.img0 = img0
        self.data_bounds = data_bounds


    def getTransfered(self):
        img = self.net.blobs['data'].data
        img = self.transformer.deprocess('data', img)
        return img

    def getLayers(self):
        return self.layers

    def getContentLayer(self):
        return self.content_layers

    def getStyleLayer(self):
        return self.style_layers

    def getNet(self):
        return self.net

    def getFcontent(self):
        return self.F_content

    def getGstyle(self):
        return self.G_style

    def getImg0(self):
        return self.img0

    def getRatio(self):
        return self.ratio

    def getBounds(self):
        return self.data_bounds

    def getStyledImg(self):
        img = self.net.blobs['data'].data
        img = self.transformer.deprocess('data', img)
        img = scipy.misc.imresize(img, (self.ori_width, self.ori_height))
        return img

