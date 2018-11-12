from numpy import *
import matplotlib.pyplot as plt
import scipy.optimize as op
from predict import predict
from mapFeatures import mapFeatures
from computeCostreg import computeCostreg
from computeGradreg import computeGradreg
from plotBoundary import plotBoundary
from sklearn.svm import SVC
import numpy as np
# # Load the dataset
# Uncomment the following command if you want to load the microchip dataset
data = loadtxt('data/microchips.csv', delimiter=',')

# Uncomment the following command in order to load bus dataset
#data = genfromtxt('bus.csv',delimiter=',')


# The first two columns contains the exam scores and the third column contains the label.
X = data[:, 0:2] 
y = data[:, 2]

# # Plot data 
plt.plot(X[:,0][y == 1], X[:,1][y == 1], 'ro', label="c1")
plt.plot(X[:,0][y == 0], X[:,1][y == 0], 'b+', label="c2")
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend(['y = 1', 'y = 0'],numpoints=1)
plt.show()

# Generate features
degree = 6
F = mapFeatures(X, degree)

# Initialize unknown parameters
w_init = zeros((F.shape[1],1))

# Regularization factor
l = 0.0

# Run minimize() to obtain the optimal coefs
w = op.fmin_bfgs(computeCostreg,w_init,args=(F, y, l),fprime=computeGradreg)

# Plot the decision boundary
plotBoundary(X, y, degree, w)
print(w)
# Compute accuracy on the training set
p = predict(array(w), F)
counter = 0
for i in range(y.size):
    if p[i] == y[i]:
        counter += 1
print('Train Accuracy: %f' % (counter / float(y.size) * 100.0))
