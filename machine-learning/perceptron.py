import numpy as np

class Perceptron(object):
	"""
	Perceptron classifier

	Parameters
	----------
	eta : float
		learning rate (must be greater that 0.0 and less than or equal to 1.0)
	n_iter : int
		the number of training times
	random_state : int
		seed for random numbers to initialize weights

	Attributes
	----------
	w_ : vector
		weights after adoption
	errors_ : list
		the number of updating in each epoch
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
		X : {Array-like data structure}, shape = {n_samples, n_features}
			Training data
			n_samples = the number of samples
			n_features = the number of features
		Y : Array-like data structure, shape = [n_samples]
			objective variable

		return
		------
		self : object

		"""
		rgen = np.random.RandomState(self.random_state)
		self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
		self.errors_ = []

		for i in range(self.n_iter):
			errors = 0
			for xi, target in zip(X, y):
				#update weights
				update = self.eta * (target - self.predict(xi))
				self.w_[1:] += update * xi
				self.w_[0] += update
				errors += int(update != 0.0)
			self.errors_.append(errors)
		return self

	def net_input(self, X):
		#calculate total input
		return np.dot(X, self.w_[1:]+self.w_[0])

	def predict(self, X):
		#return class label
		return np.where(self.net_input(X) >= 0.0, 1, -1)