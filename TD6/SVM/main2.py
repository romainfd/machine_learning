from numpy import *
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from gaussianKernel import gaussianKernel
from pylab import scatter, show, legend, xlabel, ylabel, contour, title, plot
import numpy as np

# # Load the dataset
# # The first two columns contains the exam scores and the third column
# # contains the label.
data = loadtxt('data2.txt', delimiter=',')

X = data[:, 0:2] 
y = data[:, 2]

# # Plot data 
plt.plot(X[:,0][y == 1], X[:,1][y == 1], 'r+', label="c1")
plt.plot(X[:,0][y == 0], X[:,1][y == 0], 'bo', label="c2")
plt.legend(['y = 1', 'y = 0'],numpoints=1)
plt.show()


sigma = 0.2 # Gaussian kernel variance
# We calculate the Gaussian kernel between the instances/samples
K = gaussianKernel(X,X,sigma)

C = 0.0  # SVM regularization parameter

# We create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
svc = SVC(C = C, kernel="precomputed")
svc.fit(K,y)

# Plot the decision boundary
u = linspace(min(X[:, 0]),max(X[:, 0]), 200)
v = linspace(min(X[:, 1]),max(X[:, 1]), 200)
z = zeros(shape=(len(u), len(v)))
for i in range(len(u)):
    for j in range(len(v)):
        z[i, j] = svc.predict(gaussianKernel(array([[u[i],v[j]]]),X,sigma))
        
plot(X[:,0][y == 1], X[:,1][y == 1], 'r+', label="c1")
plot(X[:,0][y == 0], X[:,1][y == 0], 'bo', label="c2")
contour(u, v, z.T, [0])
legend(['y = 1', 'y = 0', 'Decision boundary'],numpoints=1)
show()

#Compute accuracy on the training set
p = svc.predict(K)
counter = 0
for i in range(y.size):
    if p[i] == y[i]:
        counter += 1
print('Train Accuracy: %f' % (counter / float(y.size) * 100.0))
