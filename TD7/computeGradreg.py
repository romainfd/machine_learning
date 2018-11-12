from numpy import *
from sigmoid import sigmoid

def computeGradreg(theta, X, y, l, reshape=False):
    # Computes the gradient of the cost with respect to the parameters.

    m = X.shape[0]  # number of training examples
    
    grad = zeros_like(theta)  #initialize gradient

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of cost for each theta,
    # as described in the assignment.

    # we compute the usual gradient
    n = X.shape[0]
    for i in range(n):  # the influence of each training example
        delta = (sigmoid(X[i].dot(theta)) - y[i]) * transpose(X[i])
        if reshape:
            delta = delta.reshape(-1, 1)
        grad = grad + delta    # we add the regularisation cost
    grad += l * theta
    # except for theta0 that shouldn't be regularized
    grad[0] -= l * theta[0]
    # =============================================================
    return grad * 1. / m

    # =============================================================
    return grad
    
