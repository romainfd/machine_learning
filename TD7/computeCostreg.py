from numpy import *
from sigmoid import sigmoid

def computeCostreg(theta, X, y, l):
    # Computes the cost of using theta as the parameter for regularized logistic regression.
    
    m = X.shape[0] # number of training examples
    J = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Calculate the error J of the decision boundary
    #               that is described by theta  (see the assignment 
    #				for more details).
    
    for i in range(m):
        if y[i] == 1:
            J += log(sigmoid((X[i]).dot(theta)))
        else:  # == 0
            J += log(1 - sigmoid((X[i]).dot(theta)))
    J *= - 1. / m

    # we add the regularisation cost
    J += l / (2. * m) * sum(theta**2)

    # =============================================================
    return J



