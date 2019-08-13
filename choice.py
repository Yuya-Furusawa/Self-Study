import numpy as np

def sampling(coin, n):
	"""
	coin : array-like
		Amount coin
		
	n : scalar(int)
		The number of lucky guys
	"""

	coin = np.asarray(coin)
	m = len(coin)
	prob = coin / sum(coin)
	return np.random.choice(m, n, p=prob)