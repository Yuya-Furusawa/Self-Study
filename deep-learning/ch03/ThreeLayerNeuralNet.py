import numpy as np

#= Activation Functions =#

# step function
def step_function(x):
	return np.array(x > 0, dtype=np.int)

# sigmoid function
def sigmoid(x):
	return 1 / (1 + np.exp(- x))

# ReLU function
def relu(x):
	return np.maximum(0, x)

# identity function
def identity_function(x):
	return x

# soft function
def softmax(a):
	c = np.max(a)
	exp_a = np.exp(a - c) # avoid overflow
	sum_exp_a = np.sum(exp_a)
	y = exp_a / sum_exp_a
	return y



# create 3 layer neural network
def init_network():
	network = {}
	network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
	network['b1'] = np.array([0.1, 0.2, 0.3])
	network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
	network['b2'] = np.array([0.1, 0.2])
	network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
	network['b3'] = np.array([0.1, 0.2])

	return network

# forward function
# return output
def forward(network, x):
	W1, W2, W3 = network['W1'], network['W2'], network['W3']
	b1, b2, b3 = network['b1'], network['b2'], network['b3']

	a1 = np.dot(x, W1) + b1
	z1 = sigmoid(a1)
	a2 = np.dot(z1, W2) + b2
	z2 = sigmoid(a2)
	a3 = np.dot(z2, W3) + b3
	y = identity_function(a3)

	return y
