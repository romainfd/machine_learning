import numpy as np
from my_net import Network

###############################################################
# This is an example script, you may modify it as you wish
###############################################################

# Load and parse the data (N instances, D features, L=3 labels)
XY = np.genfromtxt('data/scene.csv', skip_header=1, delimiter=",")
N,DL = XY.shape
L = 6
D = DL - L
Y = XY[:,0:L].astype(int)
X = XY[:,L:D+L]

# Split into train/test sets
n = int(N*6/10)
X_train = X[0:n]
Y_train = Y[0:n]
X_test = X[n:]
Y_test = Y[n:]

from time import clock
t0 = clock()

# Test our classifier 
h = Network()

i = 0
while (clock() - t0) < 1:
    h.fit(X_train, Y_train, n_epochs=10)
    i = i + 10

print("Trained %d epochs in %d seconds." % (i, int(clock() - t0)))
Y_pred = h.predict(X_test)
print(Y_test)
print(Y_pred)
loss = np.mean(Y_pred != Y_test)
print("Hamming loss: ", loss)
