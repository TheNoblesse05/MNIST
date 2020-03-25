'''
MNIST handwritten digit classification
Data - http://yann.lecun.com/exdb/mnist/
Logistic Regression Neural Network with 3 layers
Layer 1 - 150 notdes
Layer 2 - 150 nodes
Layer 3 - 10 nodes
Author : Vedant Tilwani
'''


from __future__ import print_function
import numpy as np
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

learning_rate = 0.3
num_steps = 500
batch_size = 256
display_step = 100

#Hidden layer 1 has 150 nodes
#Hidden layer 2 has 150 nodes
#Nx is 28*28 = 784
#Final layer has 10 nodes (0-9)

X = tf.placeholder(tf.float32,[784,None],name='X')
Y = tf.placeholder(tf.float32,[10,None],name='Y')

W1 = tf.Variable(tf.random_normal([150,784],seed=1))
b1 = tf.Variable(tf.random_normal([150,1],seed=1))
W2 = tf.Variable(tf.random_normal([150,150],seed=1))
b2 = tf.Variable(tf.random_normal([150,1],seed=1))
W3 = tf.Variable(tf.random_normal([10,150],seed=1))
b3 = tf.Variable(tf.random_normal([10,1],seed=1))

Z1 = tf.add(tf.matmul(W1,X),b1)
Z2 = tf.add(tf.matmul(W2,Z1),b2)
Z3 = tf.add(tf.matmul(W3,Z2),b3)

logits = tf.transpose(Z3)
labels = tf.transpose(Y)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
train_op = optimizer

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x.T, Y: batch_y.T})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={X: batch_x.T, Y: batch_y.T})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images.T, Y: mnist.test.labels.T}))