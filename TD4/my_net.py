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

    alpha = 0.1

    def __init__(self):
        ''' initialize the classifier with default (best) parameters '''
        # TODO
        # Parameters
        self.learning_rate = 0.0005
        self.batch_size = 150
        self.display_step = 2

        # Network Parameters
        self.n_hidden_1 = 170  # 1st layer number of neurons
        self.n_hidden_2 = 80  # 2nd layer number of neurons
        self.n_input = 294  # the nb of attributes
        self.n_classes = 6  # nb of classes

        # Store layers weight & bias
        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_classes]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }
        # Data placeholders
        self.Xt = tf.placeholder("float", [None, self.n_input])
        self.Yt = tf.placeholder("float", [None, self.n_classes])
        # Construct model
        self.logits = self.multilayer_perceptron(self.Xt)
        # Define loss and optimizer
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Yt))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)
        # Initializing the variables
        self.init = tf.global_variables_initializer()
        # starts session
        self.sess = tf.Session()

    def multilayer_perceptron(self, X):
        # Hidden fully connected layer
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, self.weights['h1']), self.biases['b1']))
        # Hidden fully connected layer
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2']))
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_2, self.weights['out']) + self.biases['out']
        return out_layer

    def fit(self, X, Y, warm_start=True, n_epochs=10):
        ''' train the network, and if warm_start, then do not reinit. the network
            (if it has already been initialized)
        '''
        self.n_data = Y.shape[0]
        # TODO

        if not warm_start:
            self.sess.run(self.init)

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
            if epoch % self.display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
        print("Optimization Finished!")
        return self

    def predict_proba(self, X):
        ''' return a matrix P where P[i,j] = P(Y[i,j]=1), 
        for all instances i, and labels j. '''
        # TODO
        # all zeros: benchmark
        # return np.zeros((X.shape[0], self.n_classes))
        pred = tf.nn.softmax(self.logits)  # Apply softmax to logits to have probas
        result = self.sess.run(pred, feed_dict={self.Xt: X})
        return result

    def predict(self, X):
        ''' return a matrix of predictions for X '''
        return (self.predict_proba(X) >= 0.5).astype(int)

