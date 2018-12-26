from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

class SBS(object):
	"""
	Sequential Backward Selection
	"""
	def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
		self.scoring = scoring #criterion for evaluating features
		self.estimator = clone(estimator)
		self.k_features = k_features #the number of selected features
		self.test_size = test_size #ratio of test data
		self.random_state = random_state

	def fit(self, X, y):
		#split data into test data and train data
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
		dim = X_train.shape[1]
		self.indices_ = tuple(range(dim))
		self.subsets_ = [self.indices_]
		#calculate the scores
		score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)

		self.scores_ = [score]
		while dim > self.k_features:
			scores = []
			subsets = []
			#subsets of features
			for p in combinations(self.indices_, r=dim-1):
				score = self._calc_score(X_train, y_train, X_test, y_test, p)
				scores.append(score)
				subsets.append(p)
			#select the best score index
			best = np.argmax(scores)
			self.indices_ = subsets[best]
			self.subsets_.append(self.indices_)
			dim -= 1
			self.scores_.append(scores[best])
		#score last 
		self.k_score_ = self.scores_[-1]
		return self

	def transform(self, X):
		#return selected features
		return X[:, self.indices_]

	def _calc_score(self, X_train, y_train, X_test, y_test, indices):
		#fit the model with designated columns
		self.estimator.fit(X_train[:, indices], y_train)
		y_pred = self.estimator.predict(X_test[:, indices])
		score = self.scoring(y_test, y_pred)
		return score