# This material, largely inspired: by https://blog.keras.io/building-autoencoders-in-keras.html

from keras.layers import Input, Dense
from keras.models import Model
from numpy import loadtxt

from utils import loadMnist

# Make training and test data
n_instances = 1000
X, y = loadMnist('testing')
X_train = X[:n_instances].astype('float32')
y_train = y[:n_instances].astype('float32')
# use as a validation set
X_val = X[n_instances:].astype('float32')
y_val = y[n_instances:].astype('float32')


# this is the size of our encoded representations
encoding_dim = 2  




# TODO







# Plot clustering results
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(Z[:,0],Z[:,1], c=y)
ax.set_xlabel('1st dimension')
ax.set_ylabel('2nd dimension')
ax.set_title("Vizualization of the 2D MNIST encoding")
plt.show()
