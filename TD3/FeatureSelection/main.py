################ Feature selection for classification ######################
# This program computes the infornation gain and chisquare measures for the 
# data set "data.csv".
# Then, for a specificed number of features trains the classifier (in this 
# case logistic regression). Finally, it computes and visualizes precision/
# recall curves. 
############################################################################

import numpy as np
from math import pow
import timeit
import matplotlib.pyplot as plt
from chiSQ import chiSQ
from infogain import infogain
from logisticRegression import logisticRegression

print("\nRunning ...")

# Load the data set 
data = np.loadtxt('FeatureSelection/data.csv', delimiter=',')
#load 1st column 
Y = data[:,0:1]
# load columns 2 - end
X = data[:,1:data.shape[1]]
 
# Enables or disables feature selection
## TODO: Set 'True' or 'False' this variable
featureSelection = True

if featureSelection == True:
    # Number of features to be considered in the classification
    # should be changed to compare performance for different number 
    # of features
    ## TODO: Set the 'num_feat' variable to the number of features
    num_feat = 50

    # For each feature we get its feature selection value (x^2 or IG)
    ## TODO: uncommnent chiSQ(X,Y) to compute chi^2 measure
    #gain = infogain(X,Y)
    gain = chiSQ(X,Y)

    index = np.argsort(gain)[::-1]
    
    ########## ADD YOUR CODE HERE #######################################
    # Compute Kendall tau correlation
    ## TODO: Compute the Kendall tau correlation of the features' lists produced
    # by the two feature selection measures





    #####################################################################

    # Select the top num_feat features
    X = X[:,index[:num_feat]]
    
X = np.transpose(X)

# Start measuring execution time
start = timeit.default_timer()

# Train the classifier
w = logisticRegression(X,Y)

# Print logistic regression learning execution time
stop = timeit.default_timer()
print ('Running Time: ' + str(stop-start))

# Load test data and split data from class labels
data = np.loadtxt('FeatureSelection/test.csv', delimiter=',')
rY = data[:, 0]  # The real class labels
test = np.transpose(data[:, 1:data.shape[1]])  # the features ( 0 = label )

# Keep the features indicated by the feature selection task
if featureSelection == True: 
    test = test[index[:num_feat],:]

# Perform predictions of the test data
pY = 1/(1+np.exp(-np.dot(np.transpose(test),w)))  # predictions for the class
oldpY = np.copy(pY) # keep a copy of the class vector

# Matrix to store the precision recall values
prerec = np.zeros((101,2))

# Here we create the Precision-Recall curve applying thresholding and using
# several values for the threshold. To do so, we need to compute the confusion
# matrix (true positive, false positive, false negative)
for i in range(101):
    tp = 0 # true positive
    fp = 0 # false positive
    fn = 0 # false negative
    
    thres = float(i)/100

    pY = np.copy(oldpY)    
    
    c1 = np.where(pY>=thres)[0] # the items classified 1
    c2 = np.where(pY<thres)[0]  # the items classified 0
    pY[c1] = 1
    pY[c2] = 0
  
    # Fill confusion matrix
    for j in range(test.shape[1]):
        if pY[j] == 1:
            if rY[j] == 1: # the real class value for the specific data point
                tp += 1
            else:
                fp += 1
                
        if pY[j] == 0 and rY[j] == 1:
            fn += 1
    
    try:                 
        precisiontmp = float(tp)/(tp+fp)
    except ZeroDivisionError:
        precisiontmp = np.nan

    recalltmp = float(tp)/(tp+fn)
    
    # Fill precision-recall values for the precision-recall curve    
    prerec[i,0] = recalltmp
    prerec[i,1] = precisiontmp


# Draw the precision recall curve
plt.plot(prerec[:,0], prerec[:,1],marker='+')
print('\nTrapezoidal rule Area: ' + str(np.trapz(prerec[:,0],dx=0.01)))

if featureSelection == True:
     plt.title('Logistic regression with ' + str(num_feat) + ' features')
else:
    plt.title('Logistic regression with all features')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])    
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
        
          
