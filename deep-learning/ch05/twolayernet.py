import sys, os
sys.path.append("/Users/yuyafurusawa/deep-learning-from-scratch")
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict
from dataset.mnist import load_mnist


class TwoLayerNet:
	def __init__(self, input_size, hidden_size, output_size,
				 weight_init_std=0.01):
		"""
		Parameters
		----------
		input_size : scalar(int)
			The number of input unit
		hidden_size : scalar(int)
			The number of unit in the hidden layer
		output_size : scalar(int)
			The nuber of output unit
		weight_init_std : scalar(float)
			The standard error of initial weights (default = 0.01)
		"""
		self.params = {}
		self.params['W1'] = weight_init_std * \
							np.random.randn(input_size, hidden_size)
		self.params['b1'] = np.zeros(hidden_size)
		self.params['W2'] = weight_init_std * \
							np.random.randn(hidden_size, output_size)
		self.params['b2'] = np.zeros(output_size)

		self.layers = OrderedDict()
		self.layers['Affine1'] = \
			Affine(self.params['W1'], self.params['b1'])
		self.layers['Relu1'] = Relu()
		self.layers['Affine2'] = \
			Affine(self.params['W2'], self.params['b2'])

		self.lastLayer = SoftmaxWithLoss()

	def predict(self, x):
		"""
		Inference

		Parameters
		----------
		x : array-like
			input
		"""
		for layer in self.layers.values():
			x = layer.forward(x)

		return x

	def loss(self, x, t):
		"""
		Compute the value of loss function

		Parameters
		----------
		x : array-like
			input data
		t : array-like(ndim=1)
			true label
		"""
		y = self.predict(x)
		return self.lastLayer.forward(y, t)

	def accuracy(self, x, t):
		"""
		Compute accuracy rate

		Parameters
		----------
		x : array-like
			input data
		t : array-like
			true label
		"""
		y = self.predict(x)
		y = np.argmax(y, axis=1)
		if t.ndim != 1:
			t = np.argmax(t, axis=1)

		accuracy = np.sum(y == t) / float(x.shape[0])
		return accuracy

	def numerical_gradient(self, x, t):
		"""
		Compute gradients

		Parameters
		----------
		x : array-like
			input data
		t : array-like
			true label
		"""
		loss_W = lambda W: self.loss(x, t)

		grads = {}
		grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
		grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
		grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
		grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

		return grads

	def gradient(self, x, t):
		"""
		Compute gradients by backward propagation

		Parameters
		----------
		x : array-like
			input data
		t : array-like
			true label
		"""
		# forward
		self.loss(x, t)

		# backward
		dout = 1
		dout = self.lastLayer.backward(dout)

		layers = list(self.layers.values())
		layers.reverse()
		for layer in layers:
			dout = layer.backward(dout)

		grads = {}
		grads['W1'] = self.layers['Affine1'].dW
		grads['b1'] = self.layers['Affine1'].db
		grads['W2'] = self.layers['Affine2'].dW
		grads['b2'] = self.layers['Affine2'].db

		return grads


#= Learning =#

(x_train, t_train), (x_test, t_test) = \
	load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
	batch_mask = np.random.choice(train_size, batch_size)
	x_batch = x_train[batch_mask]
	t_batch = t_train[batch_mask]

	# compute gradient by backward propagation
	grad = network.gradient(x_batch, t_batch)

	for key in ('W1', 'b1', 'W2', 'b2'):
		network.params[key] -= learning_rate * grad[key]

	loss = network.loss(x_batch, t_batch)
	train_loss_list.append(loss)

	if i % iter_per_epoch == 0:
		train_acc = network.accuracy(x_train, t_train)
		test_acc = network.accuracy(x_test, t_test)
		train_acc_list.append(train_acc)
		test_acc_list.append(test_acc)
		print(train_acc, test_acc)
