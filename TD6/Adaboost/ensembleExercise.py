# coding: utf-8

# https://www.python.org/dev/peps/pep-0008#introduction<BR>
# http://scikit-learn.org/<BR>
# http://pandas.pydata.org/<BR>

#%%
import numpy as np
import pandas as pd
import pylab as plt

### Fetch the data and load it in pandas
data = pd.read_csv('train.csv')
print("Size of the data: ", data.shape)

#%%
# See data (five rows) using pandas tools
print(data.head())


### Prepare input to scikit and train and test cut

binary_data = data[np.logical_or(data['Cover_Type'] == 1,data['Cover_Type'] == 2)] # two-class classification set
X = binary_data.drop('Cover_Type', axis=1).values
y = binary_data['Cover_Type'].values
print(np.unique(y))
y = 2 * y - 3 # converting labels from [1,2] to [-1,1]

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
target_names = data['Cover_Type'].unique().astype(str)
target_names.sort()
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=data['Cover_Type'].unique().astype(str)))

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
# plot the lerning curves. 
# 
# Once the code is finished, play around with the hyperparameters (D and T), 
# and try to understand what is happening.

D = 2 # tree depth
T = 1000 # number of trees
w = np.ones(X_train.shape[0]) / X_train.shape[0]
training_scores = np.zeros(X_train.shape[0])
test_scores = np.zeros(X_test.shape[0])

ts = plt.arange(len(training_scores))
training_errors = []
test_errors = []

#===============================
# Your code should go here
#for t in range(T):
    
    # Your code should go here

#===============================

#  Plot training and test error
plt.plot(training_errors, label="training error")
plt.plot(test_errors, label="test error")
plt.legend()



#===================================================================
#%%
### Optimize AdaBoost

# Your final exercise is to optimize the tree depth in AdaBoost. 
# Copy-paste your AdaBoost code into a function, and call it with different tree depths 
# and, for simplicity, with T = 100 iterations (number of trees). Plot the final 
# test error vs the tree depth. Discuss the plot.

#===============================

# Your code should go here
    

#===============================


