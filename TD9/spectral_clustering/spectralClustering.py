from numpy import *
from scipy.cluster.vq import kmeans2

def spectralClustering(W, k):

    # ====================== ADD YOUR CODE HERE ======================
    # Instructions: Perform spectral clustering to partition the 
    #               data into k clusters. Implement the steps that
    #               are described in Algorithm 2 on the assignment.    

    # compute degree for each node
    # diagElems = zeros(W.shape[0])
    # for i in range(len(diagElems)):
    #    diagElems[i] = sum(W[i])  # degree of node i
    # D = diag(diagElems)
    D = diag(sum(W, axis=0))
    L = W - D
    val, U = linalg.eig(L)
    val = val.real
    U = U.real
    sortedValues = argsort(val)
    sortedVectors = U[:, sortedValues]
    minEigenVectors = sortedVectors[:, :k]
    # print(minEigenVectors.shape)  # => (450, k)

    # we apply k-means on each row
    clusters, labels = kmeans2(minEigenVectors, k)
    # =============================================================
    return labels
