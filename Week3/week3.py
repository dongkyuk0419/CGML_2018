# DongKyu Kim
# ECE 471 CGML Assignment 3
# Professor Curro

# library imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.datasets import mnist

# hyper parameters
N = 50000 # Number of training images this means 10000 of 60000 will be used for validation
learning_rate = 0.001


iteration = 50000
lambda_ = 0.0001 # regularization constant
display = 1000
sigma_noise = 0.15
samples = 2500
layers = [2,40,50,40,1] # This is my neural network set up
spiral_difficulty = 3 # This controls how many times the spiral goes around
sd = spiral_difficulty*2 # I don't have to write spiral_difficulty*2 everytime.

# Data Generation
class Data(object):
    def __init__(self,N): # N is number of training data that will be used
        (self.xtemp,self.ytemp),(self.X_test,self.Y_test) = mnist.load_data()
        mask = np.full(60000,True)
        mask[np.random.choice(60000, 60000-N, replace=False)] = False
        self.X_train = self.xtemp[mask]
        self.Y_train = self.ytemp[mask]
        self.X_val = self.xtemp[~mask]
        self.Y_val = self.ytemp[~mask]

class My_Model(object):
    def __init__(self,sess,data,learning_rate):
        self.sess = sess
        self.learning_rate
    def build_model(self):
        self.x = tf.placeholder(tf.float32,[None,28,28])
        self.y = tf.placeholder(tf.float32,[None,10])

        conv1 = tf.layer.conv2d(
            inputs = self.x,
            filters = 32,
            kernel_size = [5,5],
            padding = 'same',
            activation = tf.nn.relue)

        pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2,2], strides = 2)

        conv2 = tf.layers.conv2d(
            inputs = pool1,
            filters = 64,
            kernel_size = [5,5],
            padding = 'same',
            activation = tf.nn.relu)

        pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2,2], strides = 2)

        pool2_flat = tf.reshape(pool2,[None,7*7*64])
        dense = tf.layers.dense(inputs = pool2_flat, units = 1024, activation = tf.nn.relu)
        dropout = tf.layers.dropout(inputs = dense, rate = 0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

        self.logits = tf.layers.dense(inputs=dropout, uints=10)
    def train(self):
        self.loss = tf.loss.sparse_softmax_cross_entropy(labels = self.y, logits = self.logits)
        self.optim = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        self.init = tf.global

    def predict(self,data_test):
        temp = tf.argmax(input = self.sess.run(self.logits,feed_dict={self.x:data_test}, axis = 1),



data = Data(N)
#print(data.train_index)

print(np.shape(data.X_train[1]))
print(tf.reshape(data.X_train[0],[-1,28,28,1]))