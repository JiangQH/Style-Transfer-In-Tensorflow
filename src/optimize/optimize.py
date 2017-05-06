from tools import SimpleTools as ST
from numeric import Numeric
from scipy.optimize import minimize
import os.path as osp
import scipy.misc

class StyleTransfer(object):
	def __init__(self, content_img, style_img, model_dirs, device_id,
				 init='content', max_iter=500, verbose=True):
		self.content_img = content_img
		self.style_img = style_img
		self.model_dirs = model_dirs
		self.device_id = device_id
		self.init = init
		self.max_iter = max_iter
		self.verbose = verbose

	def transfer(self):
		"""
		Do the transfer job
		:return:
		"""
		self.tool = ST(self.model_dirs, self.content_img, self.style_img, self.device_id)
		self.optimizer = Numeric()
		# construct the optimize param
		method = 'L-BFGS-B'
		args = (self.tool.getNet(), self.tool.getLayers(), self.tool.getFcontent(),
				self.tool.getGstyle(), self.tool.getStyleLayer(), self.tool.getContentLayer(),
				self.tool.getRatio())
		jac = True
		bounds = self.tool.getBounds()
		img0 = self.tool.getImg0()
		options = {'maxiter': self.max_iter, 'disp': self.verbose}
		res = minimize(self.optimizer.computeLossAndGradAll, img0.flatten(), args=args,
					   method=method, jac=jac, bounds=bounds, options=options)
		data = self.tool.net.blobs['data'].data[0,...]
		self.tool.saveImg(data)

