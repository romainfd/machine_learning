from numpy import *
import matplotlib.pyplot as plt
from utils import loadMnist
from kNN import kNN_prediction    

# Load training and test data
X_train, y_train = loadMnist('training')
X_test, y_test = loadMnist('testing')

# Keep a subset of the training and test data
# (just to make testing faster)
X_train = X_train[:5000,:]
y_train = y_train[:5000]

# Section off some testing data
X_test = X_test[:50,:]
y_test = y_test[:50]

# Show the first ten digits (inspect the data)
fig = plt.figure('First 10 Digits') 
for i in range(10):
    a = fig.add_subplot(2,5,i+1) 
    plt.imshow(X_test[i,:].reshape(28,28), cmap=plt.cm.gray)
    plt.axis('off')
plt.show()

# Run kNN algorithm
k = int(input("Choose k, ie how many neighbors do you want to consider ? "))
y_pred = zeros(X_test.shape[0])

# Get prediction for each instance
for i in range(X_test.shape[0]):
    print("Current Test Instance: %d" % (i+1))
    y_pred[i] = kNN_prediction(k, X_train, y_train, X_test[i,:])
    
# Evaluation
accuracy = mean(y_pred == y_test)
print("\nAccuracy: %4.3f" % accuracy)
