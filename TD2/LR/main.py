from numpy import *
import matplotlib.pyplot as plt
import scipy.optimize as op

def sigmoid(z):
    # Computes the sigmoid of z.

    # ====================== YOUR CODE HERE ======================
    # Instructions: Implement the sigmoid function.
    res = 1 / (1 + exp(-z))
    # we saturate before 0 or 1 to avoid the log from taking infinite values
    if res <= 1e-4:
        res = 1e-4
    elif res >= 1 - 1e-4:
        res = 1 - 1e-4
    return res
    # =============================================================
             

def cost(theta, X, y): 
    # Computes the cost using theta as the parameters for logistic regression. 
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Calculate the error J of the decision boundary
    #               that is described by theta (see the assignment 
    #               for more details).
    cost = 0
    n = X.shape[0]
    for i in range(n):
        if y[i] == 1:
            cost += log(sigmoid((X[i]).dot(theta)))
        else:  # == 0
            cost += log(1 - sigmoid((X[i]).dot(theta)))
    return - 1. / n * cost
    # =============================================================

def compute_grad(theta, X, y):
    # Computes the gradient of the cost with respect to
    # the parameters.
    
    grad = zeros(size(theta))  # initialize gradient
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of cost for each theta.
    grad = 0
    n = X.shape[0]
    for i in range(n):  # the influence of each training example
        grad += (sigmoid((X[i]).dot(theta)) - y[i]) * transpose(X[i])
    # =============================================================
    return grad * 1./n




def predict(theta, X):
    # Predict whether each label is 0 or 1 using learned logistic 
    # regression parameters theta. The threshold is set at 0.5
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Predict the label of each instance of the
    #               training set.
    n = X.shape[0]
    preds = zeros(n)
    # we compare the proba to the threshold (1/2 here) and take the decision (either 0 or 1)
    for i in range(n):
        proba = sigmoid((X[i]).dot(theta))
        if proba >= 1./2:
            preds[i] = 1
        else:
            preds[i] = 0
    return preds
    # =============================================================
    


#======================================================================
# Load the dataset
# The first two columns contains the exam scores and the third column
# contains the label.
data = loadtxt('LR/data/data.txt', delimiter=',')
 
X = data[:, 0:2]
y = data[:, 2]

# Plot data 
pos = where(y == 1) # instances of class 1
neg = where(y == 0) # instances of class 0
plt.scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
plt.scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(['Admitted', 'Not Admitted'])
plt.title("Students admitted based on the marks at 2 exams")
plt.show()


#Add intercept term to X
X_new = ones((X.shape[0], 3))
X_new[:, 1:3] = X
X = X_new

# Initialize fitting parameters
initial_theta = random.randn(3,1) / 10


## WITH SCIPY
# Run minimize() to obtain the optimal theta
Result = op.minimize(fun = cost, x0 = initial_theta, args = (X, y), method = 'TNC',jac = compute_grad)
theta = Result.x

## WITH OUR OWN FUNCTION (stochastic gradient descent)
def optimise(x0, X, y, compute_grad, n_epochs = 1000, batch_size = 50):
    # we reshape to match the batch_size
    X_train = X.reshape(-1, batch_size, 3)
    y_train = y.reshape(-1, batch_size)
    # recursion loop
    x = x0
    for epoch in range(n_epochs):
        for i in range(X_train.shape[0]):
            x = x - (1 / (1 + epoch ** (0.5)) * compute_grad(x, X_train[i], y_train[i])).reshape(-1, 1)
    return x

# UNCOMMENT TO USE OUR OWN FUNCTION TO OPTIMIZE
# theta  = optimise(initial_theta, X, y, compute_grad, n_epochs = 50000)

# Plot the decision boundary
plot_x = array([min(X[:, 1]) - 2, max(X[:, 2]) + 2])
plot_y = (- 1.0 / theta[2]) * (theta[1] * plot_x + theta[0])
plt.plot(plot_x, plot_y)
plt.scatter(X[pos, 1], X[pos, 2], marker='o', c='b')
plt.scatter(X[neg, 1], X[neg, 2], marker='x', c='r')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(['Decision Boundary', 'Admitted', 'Not Admitted'])
plt.title("Students admitted based on the marks at 2 exams and the decision boundary computed with Linear Regression")
plt.show()

# Compute accuracy on the training set
p = predict(array(theta), X)
# Evaluation
accuracy = mean(p == y)
print("\nAccuracy: %4.3f" % accuracy)

