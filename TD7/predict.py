from numpy import *
from sigmoid import sigmoid

def predict(theta, X):
        # Predict whether the label is 0 or 1 using learned logistic
        # regression parameters theta. The threshold is set at 0.5

        m = X.shape[0] # number of training examples
	
        c = zeros(m) # predicted classes of training examples

        p = zeros(m) # logistic regression outputs of training examples

        p = sigmoid(dot(X,theta))

        for i in range(p.shape[0]):
                if p[i] > 0.5:
                        c[i] = 1
                else:
                        c[i] = 0

        return c

