from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def rbf_kernel_pca(X, gamma, n_components):
	"""
	RBF Kernel (Gauss) PCA

	Parameters
	----------
	X : {Numpy, ndarray}, shape = [n_samples, n_features]

	gamma : float
		Tuning parameter of RBF kernel

	n_components : int
		The number of returning principal components

	Returns
	-------
	X_pc : {Numpy, ndarray}, shape = [n_samples, n_features]
		projected data set

	"""
	#Compute square of euclidean distance for each pair and transrate into square matrix
	sq_dists = pdist(X, 'sqeuclidean')
	mat_sq_dists = squareform(sq_dists)

	#Compute symmetric kernel matrix
	K = exp(-gamma * mat_sq_dists)

	#Centralize kernel matrix
	N = K.shape[0]
	one_n = np.ones((N,N)) / N
	K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

	#Get eigenvectors and eigenvalues
	eigvals, eigvecs = eigh(K)
	eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

	X_pc = np.column_stack((eigvecs[:,i] for i in range(n_components)))

	return X_pc

def rbf_kernel_pca_rev(X, gamma, n_components):
	"""
	RBF Kernel (Gauss) PCA, which returns eigenvalues of kernel matrix

	Parameters
	----------
	X : {Numpy, ndarray}, shape = [n_samples, n_features]

	gamma : float
		Tuning parameter of RBF kernel

	n_components : int
		The number of returning principal components

	Returns
	-------
	alphas : {Numpy, ndarray}, shape = [n_samples, n_features]
		projected data set

	lambdas : list
		eigenvalues

	"""
	#Compute square of euclidean distance for each pair and transrate into square matrix
	sq_dists = pdist(X, 'sqeuclidean')
	mat_sq_dists = squareform(sq_dists)

	#Compute symmetric kernel matrix
	K = exp(-gamma * mat_sq_dists)

	#Centralize kernel matrix
	N = K.shape[0]
	one_n = np.ones((N,N)) / N
	K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

	#Get eigenvectors and eigenvalues
	eigvals, eigvecs = eigh(K)
	eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

	alphas = np.column_stack((eigvecs[:,i] for i in range(n_components)))

	lambdas = [eigvals[i] for i in range(n_components)]

	return alphas, lambdas