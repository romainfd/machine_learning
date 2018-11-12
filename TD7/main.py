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
from sklearn.model_selection import KFold
from tqdm import tqdm

# # Load the dataset
# Uncomment the following command if you want to load the microchip dataset
# data = loadtxt('data/microchips.csv', delimiter=',')

# Uncomment the following command in order to load bus dataset
data = genfromtxt('data/bus_train.csv', delimiter=',')

# The first two columns contains the exam scores and the third column contains the label.
X = data[:, 0:2] 
y = data[:, 2]

# # Plot data 
plt.plot(X[:,0][y == 1], X[:,1][y == 1], 'ro', label="c1")
plt.plot(X[:,0][y == 0], X[:,1][y == 0], 'b+', label="c2")
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend(['y = 1', 'y = 0'], numpoints=1)
plt.show()

# Generate features
degree = 20

# Regularization factor
l = 0.0001


# This is our home-made optimizer based on stochastic gradient descent
def optimiser(computeCost, w_init, fprime, F, y, l, batch_size=25, n_epochs=10):
    # we reshape to match the batch_size
    redNb = (F.shape[0] // batch_size) * batch_size  # to fill in the batches
    F_train = F[:redNb].reshape(-1, batch_size, F.shape[1])
    y_train = y[:redNb].reshape(-1, batch_size)
    # recursion loop
    w = w_init
    for epoch in range(n_epochs):
        for i in range(F_train.shape[0]):  # nb of batches
            delta = (1 / (1 + epoch ** 0.5) * fprime(w, F_train[i], y_train[i], l, reshape=True))
            w = w - delta
    return w


def RLR(F, X, y, degree, l, F_fold_test, y_fold_test, F_test, y_test):
    # Initialize unknown parameters
    w_init = zeros((F.shape[1], 1))
    # Run minimize() to obtain the optimal coefs
    w = op.fmin_bfgs(computeCostreg, w_init, args=(F, y, l), fprime=computeGradreg)
    # we use our own optimizer
    # w = optimiser(computeCostreg, w_init, computeGradreg, F, y, l)

    # Plot the decision boundary
    plotBoundary(X, y, degree, w, l)


    # Compute accuracy on the training set
    p = predict(array(w), F_fold_test)
    counter = 0
    for i in range(y_fold_test.size):
        if p[i] == y_fold_test[i]:
            counter += 1
    acc_train = counter / float(y_fold_test.size) * 100.0
    # computes accuracy on the test set
    p = predict(array(w), F_test)
    counter = 0
    for i in range(y_test.size):
        if p[i] == y_test[i]:
            counter += 1
    acc_test = counter / float(y_test.size) * 100.0
    return acc_train, acc_test


def cross_validation(X, degree):
    cross_validation_nb = 10
    accs = []
    ls = []
    l_cross = 1
    step_ratio = 3
    F = mapFeatures(X, degree)
    kf = KFold(n_splits=cross_validation_nb)

    # we load the test data
    test_data = genfromtxt('data/bus_test.csv', delimiter=',')
    # The first two columns contains the exam scores and the third column contains the label.
    X_test = test_data[:, 0:2]
    y_test = test_data[:, 2]
    F_test = mapFeatures(X_test, degree)

    for train_index, test_index in tqdm(kf.split(F)):
        X_train = X[train_index]
        F_train, F_fold_test = F[train_index], F[test_index]
        y_train, y_fold_test = y[train_index], y[test_index]
        accs.append(RLR(F_train, X_train, y_train, degree, l_cross, F_fold_test, y_fold_test, F_test, y_test))
        ls.append(l_cross)
        l_cross /= step_ratio
    plt.figure()
    plt.title("Cross-validation for the value of lambda")
    plt.xlabel("Value of lambda (log-scale)")
    plt.xscale("log")
    plt.ylabel("Accuracy")
    accs = np.array(accs)
    plt.plot(ls, accs[:, 0], label="Test-fold (10% of train data) accuracy")
    plt.plot(ls, accs[:, 1], label="Test data accuracy")
    plt.legend()
    plt.show()


cross_validation(X, 6)
