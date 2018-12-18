import numpy as np
from numpy.random import seed

class AdalineGD(object):
	"""
	ADAptive LInear NEuron classifier

	Parameters
	----------
	eta : float
		learning rate (must be greater than 0.0 and less than or equal yo 1.0)
	n_iter : int
		the number of training times
	random_state : int
		seed for random number to initialize weights

	Attributes
	----------
	w_ : vector
		weights after adoption
	cost_ : list
		cost function of RSS in each epoch
	"""

	def __init__(self, eta=0.01, n_iter=50, random_state=1):
		self.eta = eta
		self.n_iter = n_iter
		self.random_state = random_state

	def fit(self, X, y):
		"""
		fit for training data

		Parameters
		----------
		X : {array-like data structure}, shape = {n_samples, n_features}
			training data
			n_samples = the number of samples
			n_features = the number of features
		y : Array-like data structure, shape = [n_samples]
			objective variable

		return
		------
		self : object

		"""
		rgen = np.random.RandomState(self.random_state)
		self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
		self.cost_ = []

		for i in range(self.n_iter):
			net_input = self.net_input(X)
			output = self.activation(net_input)
			errors = y - output
			#update weights
			self.w_[1:] += self.eta * X.T.dot(errors)
			self.w_[0] += self.eta * errors.sum()
			cost = (errors**2).sum() / 2.0
			self.cost_.append(cost)
		return self

	def net_input(self, X):
		#calculate total input
		return np.dot(X, self.w_[1:]) + self.w_[0]

	def activation(self, X):
		#calculate the output of linear activation function
		#identity function
		return X

	def predict(self, X):
		#return the class label after one step
		return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


class AdalineSGD(object):
	"""
	ADAptive LInear NEuron classifier

	Parameters
	----------
	eta : float
		learning rate (must be greater than 0.0 and less than or equal yo 1.0)
	n_iter : int
		the number of training times
	shuffle : bool(default = True)
		if true, shuffle training data at each epoch to avoid circulate
	random_state : int
		seed for random number to initialize weights

	Attributes
	----------
	w_ : vector
		weights after adoption
	cost_ : list
		cost function of RSS to get average of training sample in each epoch
	"""

	def __init__(self, eta=0.01, n_iter=50, shuffle=True, random_state=None):
		self.eta = eta
		self.n_iter = n_iter
		self.w_initialized = False
		self.shuffle = shuffle
		self.random_state = random_state

	def fit(self, X, y):
		"""
		fit for training data

		Parameters
		----------
		X : {array-like data structure}, shape = {n_samples, n_features}
			training data
			n_samples = the number of samples
			n_features = the number of features
		y : Array-like data structure, shape = [n_samples]
			objective variable

		return
		------
		self : object

		"""
		self._initialize_weights(X.shape[1])
		self.cost_ = []

		for i in range(self.n_iter):
			if self.shuffle:
				X, y = self._shuffle(X,y)
			cost = []
			#calculate for each sample
			for xi, target in zip(X, y):
				cost.append(self._update_weights(xi, target))
			ave_cost = sum(cost)/len(y)
			self.cost_.append(ave_cost)
		return self

	def partial_fit(self, X, y):
		#fit for training data without re-initializing weights
		if not self.w_initialized:
			self._initialize_weights(X.shape[1])
		if y.ravel().shape[0] > 1:
			for xi, target in zip(X, y):
				self._update_weights(xi, target)
		else:
			self._update_weights(X, y)
		return self

	def _shuffle(self, X, y):
		#shuffle training data
		r = self.rgen.permutation(len(y))
		return X[r], y[r]

	def _initialize_weights(self, m):
		#initialize weights to small random number
		self.rgen = np.random.RandomState(self.random_state)
		self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1+m)
		self.w_initialized = True

	def _update_weights(self, xi, target):
		#update weights
		output = self.activation(self.net_input(xi))
		error = target - output
		self.w_[1:] += self.eta * xi.dot(error)
		self.w_[0] += self.eta * error
		cost = 0.5 * error**2
		return cost

	def net_input(self, X):
		#calculate total input
		return np.dot(X, self.w_[1:]) + self.w_[0]

	def activation(self, X):
		#calculate the output of linear activation function
		#identity function
		return X

	def predict(self, X):
		#return the class label after one step
		return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)