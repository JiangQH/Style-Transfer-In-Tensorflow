from tools import SimpleTools as ST
from numeric import Numeric
from scipy.optimize import minimize

class StyleTransfer(object):
	def __init__(self, content_img, style_img, model_name, device_id,
				 init='mixed', max_iter=500, verbose=True):
		self.content_img = content_img
		self.style_img = style_img
		self.model_name = model_name
		self.device_id = device_id
		self.init = init
		self.max_iter = max_iter
		self.verbose = verbose

	def transfer(self):
		"""
		Do the transfer job
		:return:
		"""
		self.tool = ST(self.model_name, self.content_img, self.style_img, self.device_id)
		self.optimizer = Numeric()
		# construct the optimize param
		img0 = self.tool.getImg0()
		method = 'L-BFGS-B'
		args = (self.tool.getNet(), self.tool.getLayers(), self.tool.getFcontent(),
				self.tool.getGstyle(), self.tool.getStyleLayer(), self.tool.getContentLayer(),
				self.tool.getRatio())
		jac = True
		bounds = self.tool.getBounds()
		options = {'maxiter': self.max_iter, 'disp': self.verbose}
		res = minimize(self.optimizer.computeLossAndGradAll, img0.flatten(), args=args,
					   method=method, jac=jac, bounds=bounds, options=options)
		return res

	def getStyledImg(self):
		return self.tool.getStyledImg()

