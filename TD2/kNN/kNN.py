from numpy import *
from numpy.linalg import norm

def kNN_prediction(k, X, labels, x):
    '''
    kNN classification of x
    -----------------------
        Input: 
        k: number of nearest neighbors
        X: training data           
        labels: class labels of training data
        x: test instance

        return the label to be associated with x

        Hint: you may use the function 'norm' 
    '''

    m = X.shape[0]  # number of training examples
    n = X.shape[1]  # number of attributes

    # 1. We compute all the distances
    dist = zeros(m)
    for i in range(m):
        dist[i] = norm(x - X[i])

    # 2. We find the k-nearest neighbor indices
    knn = argsort(dist)[:k]

    # 3. We determine the label of the input
    # 3.a. We count all the occurences
    occurences = zeros(max(labels) + 1)  # for [0, max]
    for neighbor_index in knn:
        neighbor_label = labels[neighbor_index]
        occurences[neighbor_label] += 1
    # 3.b. We just take the most present one between the k nearest neighbors as our label
    return argmax(occurences)

 
