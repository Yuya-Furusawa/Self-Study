import numpy as np

class LogisticRegressionGD(object):
	"""
	Rogistic regression classifier based on gradient discent

	Parameters
	----------
	eta : float
		learning rate (must be greater than 0.0 and less than or equal to 1.0)
	n_iter : int
		the number of training times
	random_state : int
		random seed to initialize weights

	Attributes
	----------
	w_ : vector
		weights after adaption
	cost_ : list
		cost function of SSR in each epoch
	"""

	def __init__(self, eta=0.05, n_iter=100, random_state=1):
		self.eta = eta
		self.n_iter = n_iter
		self.random_state = random_state

	def fit(self, X, y):
		"""
		Fit for training data

		Parameter
		---------
		X : {array-like structure}, shape = [n_samples, n_features]
			Training data
			n_samples : the number of samples
			n_features : the number of features
		y : array-like data structure, shape = [n_samples]
			objective variable

		Returns
		-------
		self : object
		"""

		rgen = np.random.RandomState(self.random_state)
		self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
		self.cost_ = []

		for i in range(self.n_iter):
			net_input = self.net_input(X)
			output = self.activation(net_input)
			errors = y - output
			self.w_[1:] += self.eta * X.T.dot(errors)
			self.w_[0] += self.eta * errors.sum()
			cost = - y.dot(np.log(output)) - ((1-y).dot(np.log(1-output)))
			self.cost_.append(cost)
		return self

	def net_input(self, X):
		#Calculate total input
		return np.dot(X, self.w_[1:]) + self.w_[0]

	def activation(self, z):
		#Calculate logistic-sigmoid activation function
		return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

	def predict(self, X):
		#Return class label one step after
		return np.where(self.net_input(X) >= 0.0, 1, 0)