import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 10
batch_size = 100

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

hidden_layer_size_1 = 160
hidden_layer_size_2 = 40
W = tf.Variable(tf.zeros([hidden_layer_size_2, 10]))
b = tf.Variable(tf.zeros([10]))
W_hidden_layer_1 = tf.Variable(tf.random_normal([784, hidden_layer_size_1]))
b_hidden_layer_1 = tf.Variable(tf.random_normal([hidden_layer_size_1]))
W_hidden_layer_2 = tf.Variable(tf.random_normal([hidden_layer_size_1, hidden_layer_size_2]))
b_hidden_layer_2 = tf.Variable(tf.random_normal([hidden_layer_size_2]))
# hidden layer
layer1 = tf.nn.softmax(tf.matmul(X, W_hidden_layer_1) + b_hidden_layer_1)
layer2 = tf.nn.softmax(tf.matmul(layer1, W_hidden_layer_2) + b_hidden_layer_2)

# prediction function
pred = tf.nn.softmax(tf.matmul(layer2, W) + b)

# compute the cost
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(pred), reduction_indices = 1))

# define the optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# init vars
init = tf.global_variables_initializer()

# start session to do computation
with tf.Session() as sess:
    sess.run(init)

    # we are doing mini-batches
    for epoch in range(20):  # nb of times we run through the data
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            # we collect our batch
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            # we run the optimisation and collect the output cost
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y})
            # the feed_dict will overwrite X, Y

            # we update the cost
            avg_cost += c / total_batch

        print("Epoch : %s, cost : %s" % (epoch + 1, avg_cost))

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # we use the test images
    print("Accuracy : " + str(accuracy.eval({X: mnist.test.images, Y: mnist.test.labels})))