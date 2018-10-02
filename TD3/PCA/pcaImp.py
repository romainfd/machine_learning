from numpy import *

#data : the data matrix
#k the number of component to return
#return the new data and  the variance that was maintained AND the principal components (ALL)
def pca(data,k):
	# Performs principal components analysis (PCA) on the n-by-p data matrix A (data)
	# Rows of A correspond to observations (wines), columns to variables.
	## TODO: Implement PCA

	# compute and substract the mean along columns
	C = data - mean(data, axis=0)
	# compute covariance matrix
	W = transpose(C).dot(C)
	# compute eigenvalues and eigenvectors of covariance matrix
	eigval, eigvec = linalg.eig(W)
	# Sort eigenvalues
	ordered_eigval_indices = eigval.argsort()[::-1]  # ::-1 to have bigger first
	# Sort eigenvectors according to eigenvalues and take top k
	Uk = eigvec[ordered_eigval_indices[:k]]
	# Project the data to the new space (k-D)
	newData = data.dot(transpose(Uk))
	# compute maintained variance
	maintained_var = 1.*sum(eigval[ordered_eigval_indices[:k]]) / sum(eigval[ordered_eigval_indices])
	return newData, maintained_var, transpose(Uk)
