import sys, os
sys.path.append("/Users/yuyafurusawa/deep-learning-from-scratch/common/")
import numpy as np
from functions import softmax, cross_entropy_error

class simpleNet:
	def __init__(self):
		"""
		Attributes
		----------
		W : array-like(shape=(2,3))
			parameters of the network
		"""
		self.W = np.random.randn(2, 3)

	def predict(self, x):
		return np.dot(x, self.W)

	def loss(self, x, t):
		"""
		x : array-like
			input
		t : array-like
			true label
		"""
		z = self.predict(x)
		y = softmax(z)
		loss = cross_entropy_error(y, t)
		return loss