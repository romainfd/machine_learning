import numpy as np
import matplotlib.pyplot as plt
from my_net import Network
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from time import clock
from tqdm import tqdm

###############################################################
# This is an example script, you may modify it as you wish
###############################################################

# Load and parse the data (N instances, D features, L=3 labels)
XY = np.genfromtxt('./TD4/data/scene.csv', skip_header=1, delimiter=",")
N, DL = XY.shape
L = 6
D = DL - L
Y = XY[:, 0:L].astype(int)
X = XY[:, L:D+L]

# Split into train/test sets
n = int(N*6/10)
X_train = X[0:n]
Y_train = Y[0:n]
X_test = X[n:]
Y_test = Y[n:]

# computes the Hamming loss
def accuracy(clsf, X_test, Y_test):
    Y_pred = clsf.predict(X_test)
    loss = np.mean(Y_pred != Y_test)
    return loss

def testClassifier(clsf, X_train, Y_train, X_test, Y_test, maxTime=3, NbEpochs=10, homeMade=False):
    t0 = clock()
    # the accuracy over the training
    accTest, accTrain, bestAcc = [], [], 1

    if homeMade:
        clsf.fit(X_train, Y_train, warm_start=False)
        accTest.append(accuracy(clsf, X_test, Y_test))
        accTrain.append(accuracy(clsf, X_train, Y_train))
        i = 0
        while (clock() - t0) < maxTime:
            clsf.fit(X_train, Y_train, n_epochs=NbEpochs)
            testAcc = accuracy(clsf, X_test, Y_test)
            if testAcc < bestAcc:
                bestAcc = testAcc
            accTest.append(testAcc)
            accTrain.append(accuracy(clsf, X_train, Y_train))
            i = i + NbEpochs
        print("Trained %d epochs in %d seconds." % (i, int(clock() - t0)))
        plt.plot(range(1, i // NbEpochs + 2), accTest, label='Test')
        plt.plot(range(1, i // NbEpochs + 2), accTrain, label='Train')
        plt.xlabel("Number of epochs (x10)")
        plt.ylabel("Hamming loss")
        plt.title("Hamming loss vs. number of epochs")
        plt.legend()
        plt.show()
    else:
        clsf.fit(X_train, Y_train)
        print("Trained in %d milliseconds." % (int(1000*(clock() - t0))))

    print("Hamming loss: {:.2%}, best loss {:.2%}".format(accuracy(clsf, X_test, Y_test), bestAcc))
    return bestAcc

# Our NN network
def testNbOfNeurons(i0, i1, step, maxTime):
    bestAccs = []
    for neurons_nb in tqdm(range(i0, i1, step)):
        h = Network(neurons_nb=[neurons_nb], display=False, scaling=False)
        acc = testClassifier(h, X_train, Y_train, X_test, Y_test, maxTime=maxTime, NbEpochs=10, homeMade=True)
        bestAccs.append(acc)
        h.sess.close()
    plt.plot(range(i0, i1, step), bestAccs)
    plt.show()
# testNbOfNeurons(10, 50, 5, 120)  # => best at 35

h = Network(neurons_nb=[35], display=False, scaling=False)
acc = testClassifier(h, X_train, Y_train, X_test, Y_test, maxTime=15, NbEpochs=10, homeMade=True)
h.sess.close()

def benchmarks():
    # sklearn.multiclass.OneVsRestClassifier
    print("------------------- OneVsRestClassifier (Naive Bayes) -------------------")
    oneVsRest = OneVsRestClassifier(MultinomialNB())
    testClassifier(oneVsRest, X_train, Y_train, X_test, Y_test)
    print("--------------- OneVsRestClassifier (Logistic Regression) ---------------")
    base_lr = LogisticRegression(solver='lbfgs')
    oneVsRest = OneVsRestClassifier(base_lr)
    testClassifier(oneVsRest, X_train, Y_train, X_test, Y_test)

    # sklearn.multiouput.ClassifierChain
    print("---------------------------- ClassifierChain ----------------------------")
    clsfChain = ClassifierChain(base_lr)
    testClassifier(clsfChain, X_train, Y_train, X_test, Y_test)

    # sklearn.neighbors.KNeighborsClassifier
    print("-------------------------- KNeighborsClassifier -------------------------")
    neigh = KNeighborsClassifier()
    testClassifier(neigh, X_train, Y_train, X_test, Y_test)

    # sklearn.tree.DecisionTreeClassifier
    print("------------------------- DecisionTreeClassfier -------------------------")
    neigh = DecisionTreeClassifier()
    testClassifier(neigh, X_train, Y_train, X_test, Y_test)

# benchmarks()