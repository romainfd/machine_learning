from numpy import *
from euclideanDistance import euclideanDistance

def findClosestNeighbours(data, N):
    
    closestNeighbours = zeros((data.shape[0], N), dtype=int32)
    distances = zeros(data.shape[0])

    # ====================== ADD YOUR CODE HERE ======================
    # Instructions: Find the N closest instances of each instance
    #               using the euclidean distance.
    for i, x in enumerate(data):
        for j, y in enumerate(data):
            distances[j] = euclideanDistance(x, y)
        # dist = 0 for himself
        closestNeighbours[i, :] = argsort(distances)[:N]
    # =============================================================
    
    return closestNeighbours
