from numpy import *
from random import randrange

def simpleInitialization(X,k):
    # Initialize k centroids randomly
    centroidIndices = []
    for i in range(k):
        r = randrange(X.shape[0])
        while r in centroidIndices:
           r = randrange(X.shape[0])
        centroidIndices.append(r)
        
    centroids = zeros((k,X.shape[1]))
    for i in range(k):
        centroids[i,:] = X[centroidIndices[i],:]
        
    return centroids
