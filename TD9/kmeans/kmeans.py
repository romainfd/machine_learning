from numpy import *
from euclideanDistance import euclideanDistance
from simpleInitialization import simpleInitialization



def kmeans(X, k):
    maxIterations = 100

    # Intialize centroids
    centroids = simpleInitialization(X, k)

    # Initialize variables
    iterations = 0
    oldCentroids = None
    labels = zeros(X.shape[0])
    
    # ====================== ADD YOUR CODE HERE ======================
    # Instructions: Run the main k-means algorithm. Follow the steps 
    #               given in the description. Compute the distance 
    #               between each instance and each centroid. Assign 
    #               the instance to the cluster described by the closest
    #               centroid. Repeat the above steps until the centroids
    #               stop moving or reached a certain number of iterations
    #               (e.g., 100).

    deltaCentroids = abs(centroids)
    while iterations < maxIterations and sum(sum(deltaCentroids)) > 0:
        # assign instances to clusters
        for i, x in enumerate(X):
            closestCluster = 0
            minDist = euclideanDistance(x, centroids[0])
            for c_i, c in enumerate(centroids[1:]):
                if euclideanDistance(x, c) < minDist:
                    minDist = euclideanDistance(x, c)
                    closestCluster = c_i + 1
            labels[i] = int(closestCluster)

        # compute new clusters
        oldCentroids = centroids.copy()
        centroids = zeros_like(centroids)
#        itemsInCluster = zeros(centroids.shape[0])
#        for i, label in enumerate(labels):
#            print(int(label))
#            itemsInCluster[int(label)] += 1
#            centroids[int(label)] += X[i]
#        for i in range(centroids.shape[0]):
#            centroids[i] /= itemsInCluster[i]
        for c in range(len(centroids)):
            centroids[c] = mean(X[labels == c], axis=0)

        deltaCentroids = abs(centroids - oldCentroids)
        iterations += 1
    # ===============================================================
    print("k-means did {} iterations".format(iterations))
    return labels
