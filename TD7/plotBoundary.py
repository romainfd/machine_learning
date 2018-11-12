from pylab import scatter, show, legend, xlabel, ylabel, contour, title, plot
from mapFeatures import mapFeatures
from numpy import *

def plotBoundary(X, y, degree, theta):

    #Plot Boundary
    u = linspace(min(X[:, 0]),max(X[:, 0]), 100)
    v = linspace(min(X[:, 1]),max(X[:, 1]), 100)
    z = zeros(shape=(len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i, j] = (mapFeatures(array([[u[i],v[j]]]),degree)).dot(theta)

    contour(u, v, z.T, [0],colors='g',linewidths=4)

    plot(X[:,0][y == 1], X[:,1][y == 1], 'ro', label="c1")
    plot(X[:,0][y == 0], X[:,1][y == 0], 'b+', label="c2")
    xlabel('Microchip Test 1')
    ylabel('Microchip Test 2')
    legend(['y = 1', 'y = 0'], numpoints = 1)
    show()
