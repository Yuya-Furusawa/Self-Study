import sys, os
sys.path.append("/Users/yuyafurusawa/deep-learning-from-scratch/")
import numpy as np
from common.functions import softmax, cross_entropy_error


class MulLayer:
	# 乗算レイヤー
	def __init__(self):
		self.x = None
		self.y = None

	def forward(self, x, y):
		self.x = x
		self.y = y
		out = x * y

		return out

	def backward(self, dout):
		dx = dout * self.y
		dy = dout * self.x

		return dx, dy

class AddLayer:
	# 加算レイヤー
	def _init__(self):
		pass

	def forward(self, x, y):
		out = x + y
		return out
	
	def backward(self, dout):
		dx = dout * 1
		dy = dout * 1

		return dx, dy

class ReluLayer:
	# ReLUレイヤ
	def __init__(self):
		self.mask = None

	def forward(self, x):
		self.mask = (x <= 0)
		out = x.copy()
		out[self.mask] = 0

		return out

	def backward(self, dout):
		dout[self.mask] = 0
		dx = dout

		return dx

class SigmoidLayer:
	# シグモイドレイヤー
	def __init__(self):
		self.out = None

	def forward(self, x):
		out = 1 / (1 + np.exp(-x))
		self.out = out

		return out

	def backward(self, dout):
		dx = dout * (1 - self.out) * self.out

		return dx

class AffineLayer:
	# Affineレイヤー
	def __init__(self, W, b):
		self.W = W
		self.b = b
		self.x = None
		self.dW = None
		self.db = None

	def forward(self, x):
		self.x = x
		out = np.dot(x, self.W) + self.b

		return out

	def backward(self, dout):
		dx = np.dot(dout, self.W.T)
		self.dW = np.dot(self.x.T, dout) # 勾配
		self.db = np.sum(dout, axis=0) # 勾配

		return dx

class SoftmaxWithLossLayer:
	# softmax + cross entropy error レイヤー
	def __init__(self):
		self.loss = None
		self.y = None
		self.t = None

	def forward(self, x, t):
		self.t = t
		self.y = softmax(x)
		self.loss = cross_entropy_error(self.y, self.t)

		return self.loss

	def backward(self, dout=1):
		batch_size = self.t.shape[0]
		dx = (self.y - self.t) / batch_size # データ１個あたりの誤差を伝播

		return dx
