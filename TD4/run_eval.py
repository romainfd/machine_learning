import numpy as np
from my_net import Network
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from time import clock

###############################################################
# This is an example script, you may modify it as you wish
###############################################################

# Load and parse the data (N instances, D features, L=3 labels)
XY = np.genfromtxt('./data/scene.csv', skip_header=1, delimiter=",")
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

def testClassifier(clsf, X_train, Y_train, X_test, Y_test, maxTime=3, NbEpochs=10, homeMade=False):
    t0 = clock()

    if homeMade:
        clsf.fit(X_train, Y_train, warm_start=False)
        i = 0
        while (clock() - t0) < maxTime:
            clsf.fit(X_train, Y_train, n_epochs=NbEpochs)
            i = i + NbEpochs
        print("Trained %d epochs in %d seconds." % (i, int(clock() - t0)))
    else:
        clsf.fit(X_train, Y_train)
        print("Trained in %d milliseconds." % (int(1000*(clock() - t0))))

    Y_pred = clsf.predict(X_test)
    #print(Y_test)
    #print(Y_pred)
    loss = np.mean(Y_pred != Y_test)
    print("Hamming loss: ", loss)

# Our NN network
h = Network()
testClassifier(h, X_train, Y_train, X_test, Y_test, maxTime=120, NbEpochs=20, homeMade=True)
h.sess.close()

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
