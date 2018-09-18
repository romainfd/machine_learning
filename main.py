from numpy import *
from matplotlib.pyplot import *
from sklearn import decomposition

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--degreeMax", type=int, default=7, help="The program will loop over degrees from 1 to maxDegree and use linear combinations of polynomial functions.")
parser.add_argument("--PCA", type=str, default='False', help="Should the program use PCA (Principal Component Analysis) ?")
args = parser.parse_args()
args.PCA = (args.PCA != 'False')

# Load the data

data = loadtxt('data/data_train.csv', delimiter=',')

# Prepare the data

X = data[:,0:-1]
y = data[:,-1]
# remove the outlier
y[y == max(y)] = 0

# Inspect the data
figure()
title('Attribute histogram')
ylabel('Average temperature')
hist(X[:,1], 10)
show()
# <TASK 1>
figure()
title('Attribute histogram')
ylabel('Average soil moisture')
hist(X[:,2], 10)
show()
figure()
title('Result histogram')
ylabel('Number of new cells')
hist(y, 10)
show()

figure()
plot(X[:, 1], X[:, 2], 'o')
title('Correlation between some attributes')
xlabel('Average temperature')
ylabel('Average soil moisture')
show()

figure()
plot(X[:, 0], y, 'o')
title('Correlation between the result and one attribute')
xlabel('Week number')
ylabel('Number of new cells')
show()

# Standardization

# <TASK 2>
if args.PCA:
    pca = decomposition.PCA(n_components=2)
    pca.fit(X)
    X = pca.transform(X)

meanX = mean(X, axis=0)
stdX = std(X, axis=0)
X = (X - meanX)/stdX

# Feature creation

from tools import poly_exp

def loop(degree):
    Z = poly_exp(X, degree)

    Z = column_stack([ones(len(Z)), Z])

    # Building a model

    # <TASK 3>
    coeffs = linalg.inv(transpose(Z).dot(Z)).dot(transpose(Z)).dot(y)
    print("coeffs = {}".format(coeffs))

    # Evaluation

    #y_pred = dot(Z_test,w)

    # <TASK 4>
    data_test = loadtxt('data/data_test.csv', delimiter=',')

    # Prepare the data
    X_test = data_test[:, 0:-1]
    if args.PCA:
        X_test = pca.transform(X_test)

    y_test = data_test[:, -1]
    X_test = (X_test - meanX)/stdX

    Z_test = poly_exp(X_test, degree)
    Z_test = column_stack([ones(len(Z_test)), Z_test])

    from tools import MSE

    # <TASK 5>
    # <TASK 6>
    msePred = MSE(y_test, Z_test.dot(coeffs))
    print("MSE on test data   {}".format(msePred))
    mseBaseline = MSE(y_test, mean(y)*np.ones(len(y_test)))
    print("MSE baseline       {}".format(mseBaseline))
    mseTrain = MSE(y, Z.dot(coeffs))
    return msePred, mseTrain


# <TASK 7>
def overfit(degreeMax):
    test_mse = np.zeros(degreeMax)
    train_mse = np.zeros(degreeMax)
    for degree in range(0, degreeMax):
        msePred, mseTrain = loop(degree + 1)
        test_mse[degree] = msePred
        train_mse[degree] = mseTrain
    figure()
    title('Example of overfitting by using high degree polynomial functions')
    plot(range(1, degreeMax + 1), test_mse, label='test')
    plot(range(1, degreeMax + 1), train_mse, label='train')
    xlabel('Degree')
    ylabel('MSE')
    legend()
    show()

# <TASK 8: You will need to make changes from '# Feature creation'
#          To get the exact results, you will need to reverse the second part of Task 7 (your own modifications)>

overfit(args.degreeMax)