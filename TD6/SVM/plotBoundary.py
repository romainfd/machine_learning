from pylab import scatter, show, legend, xlabel, ylabel, contour, title, plot
from sklearn.svm import SVC
from numpy import *
from gaussianKernel import gaussianKernel

def plotBoundary(X, y, svm):

    #Plot Boundary
    u = linspace(min(X[:, 0]),max(X[:, 0]), 200)
    v = linspace(min(X[:, 1]),max(X[:, 1]), 200)
    z = zeros(shape=(len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i, j] = svm.predict(gaussianKernel(array([[u[i],v[j]]]),X))
                                  
    plot(X[:,0][y == 1], X[:,1][y == 1], 'ro', label="c1")
    plot(X[:,0][y == 0], X[:,1][y == 0], 'b+', label="c2")
    contour(u, v, z.T, [0])
    xlabel('Microchip Test 1')
    ylabel('Microchip Test 2')
    legend(['y = 1', 'y = 0', 'Decision boundary'],numpoints=1)
    show()
