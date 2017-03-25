# MNIST_CNN

The objective of this project is to implement a Convolutional Neural Network using Tensorflow for predicting MNIST images.

The MNIST dataset is downloaded from http://yann.lecun.com/exdb/mnist/ (four files) and extracted into a folder named 'data' just outside the folder containing the main.py file.
That is, the code reads the input data files from the folder '../data'. The train function in train_cnn.py trains the neural network given the training examples and saves the weights in a folder named 'weights_cnn' in the same folder as main.py. 
The `test` function reads the saved weights and given the test examples it returns the predicted labels.

