weights_for_test_location = "./weights_cnn/model1_cnn.ckpt"

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os

#function to create mini_batches for training
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]

    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)

    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        yield inputs[excerpt], targets[excerpt]

#function to initialize a variable of type 'weight'
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

#function to initialize a variable of type 'bias'
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#function for implementing Convolution operation
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

#function for implementing Max Pooling operation
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

#function to train the CNN
def train(trainX, trainY):
    # Parameters
    learning_rate = 0.0005
    regularization_rate = 0.02
    training_epochs = 3
    batch_size = 150
    display_step = 1

    # Network Parameters
    n_input = 784
    n_classes = 10

    num_training_examples = trainX.shape[0]

    trainX = trainX.reshape(num_training_examples, n_input)
    trainY = trainY.reshape(num_training_examples, 1)

    trainY_actual = np.zeros([num_training_examples, n_classes])
    for pos, val in enumerate(trainY):
        trainY_actual[pos, val] = 1

    #Weight Initialization Functions
    x = tf.placeholder("float", shape = [None, n_input])
    y_ = tf.placeholder("float", shape = [None, n_classes])

    x = tf.reshape(x, [-1, 28, 28, 1])      #Make image a 4D tensor

    #1st pair of Convolutional and Max Pool Layers
    W_conv1 = weight_variable([5, 5, 1, 8])
    b_conv1 = bias_variable([8])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    #2nd pair of Convolutional and Max Pool Layers
    W_conv2 = weight_variable([5, 5, 8, 16])
    b_conv2 = bias_variable([16])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #Layer for flattening the output of 2nd Max Pool layer
    flat_shape = h_pool2.get_shape()
    flat_feature_size = flat_shape[1:4].num_elements()
    h_pool2_flat = tf.reshape(h_pool2, [-1, flat_feature_size])

    #Fully connected layer
    W_fcl = weight_variable([flat_feature_size, 256])
    b_fcl = bias_variable([256])
    h_fcl = tf.nn.relu(tf.matmul(h_pool2_flat, W_fcl) + b_fcl)

    #Output Layer (softmax)
    W_fc2 = weight_variable([256, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fcl, W_fc2) + b_fc2)

    #Train and Evaluate the Model
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_)) + (regularization_rate*tf.nn.l2_loss(W_conv1)) + (regularization_rate*tf.nn.l2_loss(W_conv2)) + (regularization_rate*tf.nn.l2_loss(W_fcl)) + (regularization_rate*tf.nn.l2_loss(W_fc2))
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
    # correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver({'W_conv1':W_conv1, 'b_conv1':b_conv1, 'W_conv2':W_conv2, 'b_conv2':b_conv2, 'W_fcl':W_fcl, 'b_fcl':b_fcl, 'W_fc2':W_fc2, 'b_fc2':b_fc2})

        for epoch in range(training_epochs):
            # avg_cost = float(0.)
            # total_number_of_batches = int(training_data_Y.shape[0] / batch_size)

            for batch in iterate_minibatches (trainX, trainY_actual, batch_size, shuffle=True):
                batch_x, batch_y = batch
                sess.run(train_step, feed_dict = {x: batch_x.reshape([batch_size,28,28,1]), y_: batch_y})  # Run optimization op (backprop) and cost op (to get loss value)

            if epoch % display_step == 0:
                print("Current epoch : ", (epoch+1))

        saver.save(sess, './weights_cnn/model1_cnn.ckpt')
        print "Training complete"

#function to test the trained CNN model
def test(testX):
    '''
    This function reads the weight files and
    return the predicted labels.
    The returned object is a 1-dimensional numpy array of
    length equal to the number of examples. The i-th element
    of the array contains the label of the i-th test
    example.
    '''
    # Network Parameters

    n_input = 784
    n_classes = 10

    #Weight Initialization Functions
    x = tf.placeholder("float", shape = [None, n_input])
    y_ = tf.placeholder("float", shape = [None, n_classes])

    x = tf.reshape(x, [-1, 28, 28, 1])      #Make image a 4D tensor

    #1st pair of Convolutional and Max Pool Layers
    W_conv1 = weight_variable([5, 5, 1, 8])
    b_conv1 = bias_variable([8])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    #2nd pair of Convolutional and Max Pool Layers
    W_conv2 = weight_variable([5, 5, 8, 16])
    b_conv2 = bias_variable([16])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #Layer for flattening the output of 2nd Max Pool layer
    flat_shape = h_pool2.get_shape()
    flat_feature_size = flat_shape[1:4].num_elements()
    h_pool2_flat = tf.reshape(h_pool2, [-1, flat_feature_size])

    #Fully connected layer
    W_fcl = weight_variable([flat_feature_size, 256])
    b_fcl = bias_variable([256])
    h_fcl = tf.nn.relu(tf.matmul(h_pool2_flat, W_fcl) + b_fcl)

    #Output Layer (softmax)
    W_fc2 = weight_variable([256, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fcl, W_fc2) + b_fc2)

    with  tf.Session() as sess:
        saver = tf.train.Saver({'W_conv1':W_conv1, 'b_conv1':b_conv1, 'W_conv2':W_conv2, 'b_conv2':b_conv2, 'W_fcl':W_fcl, 'b_fcl':b_fcl, 'W_fc2':W_fc2, 'b_fc2':b_fc2})
        saver.restore(sess, weights_for_test_location)
        output_values = sess.run(y_conv, feed_dict={x:testX})

    print("Testing Finished.")

    predicted_labels = np.argmax(output_values, axis=1)

    return predicted_labels
