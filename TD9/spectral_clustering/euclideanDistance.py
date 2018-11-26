from numpy import *

def euclideanDistance(vectorA, vectorB):
	# Computes the euclidean distance between two vectors. The
	# two vectors must have the same size.

	mA = len(vectorA) # length of vectorA
	mB = len(vectorB) # length of vectorB
	
	assert mA == mB, 'The two vectors must have the same size'
	
	distance = 0
	
	for i in range(mA):
		distance = distance + pow((vectorA[i]-vectorB[i]),2)
		
	distance = sqrt(distance)
	
	return distance