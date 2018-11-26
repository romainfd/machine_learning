from numpy import *

def euclideanDistance(vectorA, vectorB):

    mA = len(vectorA)
    mB = len(vectorB)

    assert mA == mB, 'The two vectors must have the same size'

    distance = 0

    for i in range(mA):
        distance = distance + pow((vectorA[i]-vectorB[i]),2)

    distance = sqrt(distance)
    
    return distance
