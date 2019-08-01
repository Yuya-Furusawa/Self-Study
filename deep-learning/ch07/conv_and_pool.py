import sys
sys.path.append("Users/yuyafurusawa/deep-learning-from-scratch/")
import numpy as np
from common.util import im2col


#= Convolution Layer =#
class Convolution:
	def __init__(self, W, b, stride=1, pad=0):
		self.W = W
		self.b = b
		self.stride = stride
		self.pad = pad

	def forward(self, x):
		FN, C, FH, FW = self.W.shape
		N, C, H, W = x.shape
		out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
		out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

		col = im2col(x, FH, FW, self.stride, self.pad)
		col_W = self.W.reshape(FN, -1).T
		out  = np.dot(col, col_W) + self.b

		out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

		return out


#= Pool Layer =#
class Pooling:
	def __init__(self, pool_h, pool_w, stride=1, pad=0):
		self.pool_h = pool_h
		self.pool_w = pool_w
		self.stride = stride
		self.pad = pad

	def forward(self, x):
		N, C, H, W = x.shape
		out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
		out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

		# reshape to 2-dimensional array
		col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
		col = col.reshape(-1, self.pool_h*self.pool_w)

		out = np.max(col, axis=1)
		# reshape again
		out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

		return out