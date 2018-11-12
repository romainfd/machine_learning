# coding: utf-8

# https://www.python.org/dev/peps/pep-0008#introduction<BR>
# http://scikit-learn.org/<BR>
# http://pandas.pydata.org/<BR>

#%%
import numpy as np
import pandas as pd
import pylab as plt

### Fetch the data and load it in pandas
data = pd.read_csv('./Adaboost/train.csv')
print("Size of the data: ", data.shape)

#%%
# See data (five rows) using pandas tools
print(data.head())


### Prepare input to scikit and train and test cut
# we focus only on the binary pbm (class = 1 or 2)
binary_data = data[np.logical_or(data['Cover_Type'] == 1, data['Cover_Type'] == 2)]  # two-class classification set
X = binary_data.drop('Cover_Type', axis=1).values
y = binary_data['Cover_Type'].values
print(np.unique(y))  # we should only have classes 1 and 2
y = 2 * y - 3  # converting labels from [1,2] to [-1,1]

#%%
# Import cross validation tools from scikit
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=None)


#%%
### Train a single decision tree

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=8)

# Train the classifier and print training time
clf.fit(X_train, y_train)

#%%
# Do classification on the test dataset and print classification results
from sklearn.metrics import classification_report
target_names = binary_data['Cover_Type'].unique().astype(str)
target_names.sort()
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=binary_data['Cover_Type'].unique().astype(str)))

#%%
# Compute accuracy of the classifier (correctly classified instances)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

#===================================================================
#%%
### Train AdaBoost

# Your first exercise is to program AdaBoost.
# You can call *DecisionTreeClassifier* as above, 
# but you have to figure out how to pass the weight vector (for weighted classification) 
# to the *fit* function using the help pages of scikit-learn. At the end of 
# the loop, compute the training and test errors so the last section of the code can 
# plot the learning curves.
# 
# Once the code is finished, play around with the hyperparameters (D and T), 
# and try to understand what is happening.

#===============================
# Your code should go here
def adaboost(D=8, T=100, display=True):
    w = np.ones(X_train.shape[0]) * 1. / X_train.shape[0]
    training_scores = np.zeros(X_train.shape[0])
    test_scores = np.zeros(X_test.shape[0])

    ts = plt.arange(len(training_scores))
    training_errors = []
    test_errors = []
    for t in range(T):
        # we train the weak classifier on the weighted sample
        clf = DecisionTreeClassifier(max_depth=D)
        clf.fit(X_train, y_train, sample_weight=w)

        # we add the results our new classifier to the final classifier
        y_pred = clf.predict(X_train)
        gamma = np.sum(w[np.where(y_pred != y_train)]) / np.sum(w)
        alpha = np.log((1 - gamma) / gamma)
        training_scores += alpha * y_pred
        y_pred_test = clf.predict(X_test)
        test_scores += alpha * y_pred_test

        # We update the weights based on the results of this classifier
        w = w * np.exp(alpha * (y_pred != y_train))

        # Compute the errors to plot the learning curves
        train_error = np.mean(y_train != np.sign(training_scores))
        training_errors.append(train_error)
        test_error = np.mean(y_test != np.sign(test_scores))
        test_errors.append(test_error)

    if display:
        #  Plot training and test error
        plt.plot(training_errors, label="training error")
        plt.plot(test_errors, label="test error")
        plt.title("Adaboost based on {} trees of depth {}".format(T, D))
        plt.legend()
        plt.show()
    return test_errors[-1]
#===============================
adaboost(D=8, T=100, display=True)

#===================================================================
#%%
### Optimize AdaBoost

# Your final exercise is to optimize the tree depth in AdaBoost. 
# Copy-paste your AdaBoost code into a function, and call it with different tree depths 
# and, for simplicity, with T = 100 iterations (number of trees). Plot the final 
# test error vs the tree depth. Discuss the plot.

#===============================

# Your code should go here
errors = []
for depth in range(1, 18):
    errors.append(adaboost(D=depth, T=100, display=False))
plt.plot(range(1, 18), errors)
plt.title("Test error with 100 decisions trees depending on the depth")
plt.xlabel("Depth of the decision trees")
plt.show()
#===============================


