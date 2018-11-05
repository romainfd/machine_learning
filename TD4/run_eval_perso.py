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
import seaborn as sns
import pandas as pd
import tensorflow as tf

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
        i = 0
        while (clock() - t0) < maxTime:
            clsf.fit(X_train, Y_train, n_epochs=NbEpochs)
            testAcc = accuracy(clsf, X_test, Y_test)
            if testAcc < bestAcc:
                bestAcc = testAcc
                print("Best time: {}".format(int(1000*(clock() - t0))))
            accTest.append(testAcc)
            accTrain.append(accuracy(clsf, X_train, Y_train))
            i = i + NbEpochs
        print("Trained %d epochs in %d seconds." % (i, int(clock() - t0)))
        plt.plot(range(1, i // NbEpochs + 1), accTest, label='Test')
        plt.plot(range(1, i // NbEpochs + 1), accTrain, label='Train')
        plt.xlabel("Number of epochs (x10)")
        plt.ylabel("Hamming loss")
        plt.title("Hamming loss vs. number of epochs (batch_size="+str(clsf.nns[0].batch_size)+")")
        plt.legend()
        plt.show()
    else:
        clsf.fit(X_train, Y_train)
        print("Trained in %d milliseconds." % (int(1000*(clock() - t0))))

    print("Hamming loss: {:.2%}, best loss {:.2%}".format(accuracy(clsf, X_test, Y_test), bestAcc))
    return bestAcc
    #return accTest  # for testNbLayers

# Our NN network
def testNbOfNeurons(i0, i1, step, maxTime=120):
    print("Expected time: {} minutes".format((i1 - step - i0)//step * maxTime // 60))
    bestAccs = []
    for neurons_nb in tqdm(range(i0, i1, step)):
        tf.set_random_seed(2106)
        h = Network(neurons_nb=[neurons_nb], display=False, scaling=False)
        acc = testClassifier(h, X_train, Y_train, X_test, Y_test, maxTime=maxTime, NbEpochs=10, homeMade=True)
        bestAccs.append(acc)
        h.close()
    plt.title("Best accuracy obtained vs. the number of neurons in the hidden layer")
    plt.xlabel("Number of neurons in the hidden layer")
    plt.ylabel("Best accuracy obtained over a 2 minutes training")
    plt.plot(range(i0, i1, step), bestAccs)
    plt.show()
# testNbOfNeurons(5, 200, 5, maxTime=120)  # => best at 35
#testNbOfNeurons(5, 10, 5, maxTime=5)  # => best at 35

def testBagging(bag_nbs, bag_ratios, maxTime=120):
    print("Expected time: {} minutes".format(len(bag_nbs) * len(bag_ratios) * maxTime // 60))
    accs = pd.DataFrame(columns=bag_ratios, index=bag_nbs, dtype=float)
    for nb in tqdm(bag_nbs):
        for r in bag_ratios:
            np.random.seed(seed=0)
            tf.set_random_seed(2106)
            h = Network(neurons_nb=[70], display=False, scaling=False, bagging=True, bag_nb=nb, bag_ratio=r)
            accs.loc[nb, r] = testClassifier(h, X_train, Y_train, X_test, Y_test, maxTime=maxTime, NbEpochs=10, homeMade=True)
            h.close()
    print(accs)
    sns.heatmap(accs, annot=True, fmt=".1%")
    plt.rc('text', usetex=True)
    plt.title("Heatmap of the best accuracy obtained \nvs. the bagging ratio and the number of bags ($h_1$=35, batch_size=150)")
    plt.xlabel("Bag ratio")
    plt.ylabel("Number of bags")
    plt.show()

# testBagging([1, 2, 3, 4, 7, 10], [0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10], maxTime=120)  # => best = 4, 4
#testBagging([1], [0.25, 0.25, 0.25], maxTime=5)

def testBatchSize(i0, i1, step, maxTime=120):
    print("Expected time: {} minutes".format((i1 - step - i0)//step * maxTime // 60))
    bestAccs = []
    for batchSize in tqdm(range(i0, i1, step)):
        tf.set_random_seed(2106)
        np.random.seed(seed=0)
        h = Network(display=False, scaling=False, batch_size=int(batchSize))
        acc = testClassifier(h, X_train, Y_train, X_test, Y_test, maxTime=maxTime, NbEpochs=10, homeMade=True)
        bestAccs.append(acc)
        h.close()
    plt.title("Best accuracy obtained vs. the batch_size of the mini-batches")
    plt.xlabel("Batch_size")
    plt.ylabel("Best accuracy obtained over a 2 minutes training")
    plt.plot(range(i0, i1, step), bestAccs)
    plt.show()


# testBatchSize(5, 200, 5, maxTime=120)

def singleTest(batch_size=65, neurons_nb=[130], bagging=True, bag_nb=4, bag_ratio=4, maxTime=120):
    tf.set_random_seed(2106)
    np.random.seed(seed=0)
    h0 = Network(display=True, scaling=False, batch_size=batch_size, neurons_nb=neurons_nb, bagging=bagging, bag_nb=bag_nb, bag_ratio=bag_ratio)
    acc = testClassifier(h0, X_train, Y_train, X_test, Y_test, maxTime=maxTime, NbEpochs=10, homeMade=True)
    h0.close()


#singleTest(batch_size=70, neurons_nb=[130], bagging=True, bag_nb=4, bag_ratio=4, maxTime=120)

def defaultTest():
    tf.set_random_seed(2106)
    np.random.seed(seed=0)
    h0 = Network()
    acc = testClassifier(h0, X_train, Y_train, X_test, Y_test, maxTime=120, NbEpochs=10, homeMade=True)
    h0.close()

# defaultTest()

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

benchmarks()

def heatmap():
    accs = pd.DataFrame(
        data=[[.12, .108, .11, .118, .179],
              [.108, .095, .094, .094, .142],
              [.096, .086, .083, .082, .098],
              [.09, .081, .082, .082, .092],
              [.095, .082, .081, .078, .082],
              [.09, .082, .079, .082, .084],
              [.091, .081, .081, .081, .084],
              [.09, .08, .08, .08, .082]],
        columns=[1, 2, 3, 4, 7],
        index=[0.25, 0.5, 1, 2, 4, 6, 8, 10],
        dtype=float)
    print(accs)
    sns.heatmap(accs, annot=True, fmt=".1%", vmin=0.078, vmax=0.12, cmap="RdYlGn_r")
    plt.rc('text', usetex=True)
    plt.title("Heatmap of the best accuracy obtained \nvs. the bagging ratio and the number of bags ($h_1$=70, batch_size=150)")
    plt.ylabel("Bag ratio")
    plt.xlabel("Number of bags")
    plt.show()

# heatmap()

def testNbLayers(batch_size=150, maxTime=120):
    costs = []
    accs = []
    for neurons_nb in ([], [50], [150, 75], [300, 200, 100]):
        tf.set_random_seed(2106)
        np.random.seed(seed=0)
        h0 = Network(display=True, scaling=False, batch_size=batch_size, neurons_nb=neurons_nb, bagging=False)
        acc = testClassifier(h0, X_train, Y_train, X_test, Y_test, maxTime=maxTime, NbEpochs=10, homeMade=True)
        accs.append(acc)
        costs.append(h0.nns[0].costs)
        h0.close()
    plt.figure()
    for i, cost in enumerate(costs):
        plt.plot(range(1, len(cost) + 1), cost, label='{} layer(s)'.format(i))
    plt.xlabel("Number of epochs (x10)")
    plt.ylabel("Cross-entropy loss on an entire epoch")
    plt.title("Loss for different layer configurations (batch_size=" + str(batch_size) + ")")
    plt.legend()
    plt.show()
    plt.figure()
    for i, acc in enumerate(accs):
        plt.plot(range(1, len(acc) + 1), acc, label='{} layer(s)'.format(i))
    plt.xlabel("Number of epochs (x10)")
    plt.ylabel("Hamming loss")
    plt.title("Hamming loss on the test data \nfor different layer configurations (batch_size=" + str(batch_size) + ")")
    plt.legend()
    plt.show()

# testNbLayers(maxTime=120)