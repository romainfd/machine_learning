import numpy as np
import tensorflow as tf

###############################################################
#
# Important notes: 
# - Do not change any of the existing functions or parameter names, 
#       except in __init__, you may add/change parameter names/defaults values.
# - In __init__ set default values to the best ones, e.g., learning_rate=0.1
# - Training epochs/iterations should not be a parameter to __init__,
#   To train/test your network, we will call fit(...) until time (2 mins) runs out.
#
###############################################################


class Network():
    def __init__(self, neurons_nb=[115], display=False, scaling=False, bagging=True, bag_nb=4, bag_ratio=4, batch_size=100):
        self.bagging = bagging
        self.nns = []
        if bagging:
            self.bag_nb = bag_nb
            self.bag_ratio = bag_ratio
            self.bags = []
            for i in range(bag_nb):
                self.nns.append(Bag(neurons_nb=neurons_nb, display=display, scaling=scaling, batch_size=batch_size))
                self.bags.append([])
        else:
            self.nns = [Bag(neurons_nb=neurons_nb, display=display, scaling=scaling, batch_size=batch_size)]
        self.warm_start = False

    def fit(self, X, Y, n_epochs=10):
        ''' train the network, and if warm_start, then do not reinit. the network
            (if it has already been initialized)
        '''
        if not self.warm_start:
            # we compute the bags
            if self.bagging:
                for i in range(int(self.bag_nb)):
                    for _ in range(int(X.shape[0] * self.bag_ratio)):
                        self.bags[i].append(int(X.shape[0] * np.random.random()))  # the index of the item
            else:
                self.bags = [list(range(X.shape[0]))]  # we take all the items once
            self.warm_start = True

        for i, nn in enumerate(self.nns):
            nn.fit(X[self.bags[i]], Y[self.bags[i]], n_epochs=n_epochs)
        return self

    def predict_proba(self, X):
        ''' return a matrix P where P[i,j] = P(Y[i,j]=1),
        for all instances i, and labels j. '''
        # TODO
        result = np.zeros((X.shape[0], self.nns[0].neurons_nb[-1]))  # the nb of classes
        for i, nn in enumerate(self.nns):
            pred = tf.nn.softmax(nn.logits)  # Apply softmax to logits to have probas
            if nn.scaling:
                X = (X - nn.X_mean) / nn.X_std
            result += nn.sess.run(pred, feed_dict={nn.Xt: X})
        return result / len(self.nns)

    def predict(self, X):
        ''' return a matrix of predictions for X '''
        return (self.predict_proba(X) >= 0.5).astype(int)

    def close(self):
        for nn in self.nns:
            nn.sess.close()
        tf.reset_default_graph()


class Bag():
    def __init__(self, neurons_nb=[130], display=False, scaling=True, batch_size=70):
        ''' initialize the classifier with default (best) parameters '''
        # TODO
        # Parameters
        self.learning_rate = 0.0001
        self.batch_size = batch_size
        self.display_step = 2
        self.display = display

        # Network Parameters
        self.n_classes = 6  # nb of classes
        self.neurons_nb = [294]  # nb inputs
        self.weights = {}
        self.biases = {}
        # neurons_nb.append(6)  # nb outputs
        for i in range(1, 1 + len(neurons_nb)):
            self.neurons_nb.append(neurons_nb[i - 1])
            # Store layers weight & bias
            self.weights["h"+str(i)] = tf.Variable(tf.random_normal([self.neurons_nb[i - 1], self.neurons_nb[i]]))
            self.biases["b"+str(i)] = tf.Variable(tf.random_normal([self.neurons_nb[i]]))

        # output layer
        self.neurons_nb.append(6)
        # Store layers weight & bias
        self.weights["h" + str(1 + len(neurons_nb))] = tf.Variable(tf.random_normal([self.neurons_nb[1 + len(neurons_nb) - 1], self.neurons_nb[1 + len(neurons_nb)]]))
        self.biases["b" + str(1 + len(neurons_nb))] = tf.Variable(tf.random_normal([self.neurons_nb[1 + len(neurons_nb)]]))

        # Data placeholders
        self.Xt = tf.placeholder("float", [None, self.neurons_nb[0]])  # nb inputs
        self.Yt = tf.placeholder("float", [None, self.neurons_nb[-1]])  # nb of classes
        self.scaling = scaling
        if scaling:
            self.X_mean = []
            self.X_std = []
        # Construct model
        self.logits = self.multilayer_perceptron(self.Xt)
        # Define loss and optimizer
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Yt))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)
        # Initializing the variables
        self.init = tf.global_variables_initializer()
        # starts session
        self.sess = tf.Session()
        # but not initialized yet
        self.warm_start = False

        # cost over time
        self.costs = []

    def multilayer_perceptron(self, X):
        previous_layer = X  # we start with the inputs
        for i in range(1, len(self.neurons_nb) - 1):  # we don't take the input and output layers
            previous_layer = tf.nn.sigmoid(tf.add(tf.matmul(previous_layer, self.weights['h' + str(i)]), self.biases['b' + str(i)]))
        # Output fully connected layer with a neuron for each class
        last = len(self.neurons_nb) - 1
        out_layer = tf.matmul(previous_layer, self.weights['h' + str(last)]) + self.biases['b' + str(last)]
        return out_layer

    def fit(self, X, Y, n_epochs=10):
        ''' train the network, and if warm_start, then do not reinit. the network
            (if it has already been initialized)
        '''
        self.n_data = Y.shape[0]
        # TODO

        if not self.warm_start:
            self.sess.run(self.init)
            if self.scaling:
                self.X_mean = np.mean(X, axis=0)
                self.X_std = np.std(X, axis=0)
            self.warm_start = True

        # auto scale
        if self.scaling:
            X = (X - self.X_mean) / self.X_std

        # Training cycle
        for epoch in range(n_epochs):
            avg_cost = 0.
            total_batch = int(self.n_data / self.batch_size)
            # Loop over all batches
            for i in range(total_batch - 1):
                batch_x = X[i * self.batch_size: (i + 1) * self.batch_size, :]
                batch_y = Y[i * self.batch_size: (i + 1) * self.batch_size, :]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = self.sess.run([self.train_op, self.loss_op], feed_dict={self.Xt: batch_x, self.Yt: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % self.display_step == 0 and self.display:
                print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
        self.costs.append(avg_cost)
        return self

    def predict_proba(self, X):
        ''' return a matrix P where P[i,j] = P(Y[i,j]=1), 
        for all instances i, and labels j. '''
        # TODO
        # all zeros: benchmark
        # return np.zeros((X.shape[0], self.n_classes))
        pred = tf.nn.softmax(self.logits)  # Apply softmax to logits to have probas
        if self.scaling:
            X = (X - self.X_mean) / self.X_std
        result = self.sess.run(pred, feed_dict={self.Xt: X})
        return result

    def predict(self, X):
        ''' return a matrix of predictions for X '''
        return (self.predict_proba(X) >= 0.5).astype(int)



