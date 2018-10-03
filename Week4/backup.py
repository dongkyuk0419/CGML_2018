# DongKyu Kim
# ECE 471 CGML Assignment 4
# CIFAR-10
# Professor Curro

# library imports
import numpy as np
from tqdm import tqdm
from keras import optimizers, regularizers
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization


# Parameters
r = 0.9 # Ratio of training data otu of training + validation
batch_size = 128
epochs = 32
numclass = 10

# hyper parameters
learning_rate = 1e-3
filters = [32,32,64,64]
kernel_size = [3,3]
lambda_ = 1e-5 # regularization constant

# Useful functions
def one_hot_encoding(data,numclass):
    targets = np.array(data).reshape(-1)
    return np.eye(numclass)[targets]

# Data Generation
class Data(object):
    def __init__(self,r,numclass): # N is number of training data that will be used
        (self.xtemp,self.ytemp),(self.X_test,self.Y_test) = cifar10.load_data()
        mask = np.full(self.xtemp.shape[0],True)
        mask[np.random.choice(self.xtemp.shape[0], int(self.xtemp.shape[0]*(1-r)), replace=False)] = False
        self.numclass = numclass
        self.X_train = self.xtemp[mask].astype('float32')
        self.Y_train = self.ytemp[mask]
        self.X_val = self.xtemp[~mask].astype('float32')
        self.Y_val = self.ytemp[~mask]
        self.X_test = self.X_test.astype('float32')
        self.Y_test = self.Y_test
        self.X_train /= 255
        self.X_val /= 255
        self.X_test /= 255
        self.Y_train = one_hot_encoding(self.Y_train,self.numclass)
        self.Y_val = one_hot_encoding(self.Y_val,self.numclass)
        self.Y_test = one_hot_encoding(self.Y_test,self.numclass)

# Model
class My_Model(object):
    def __init__(self,data,learning_rate,filters,kernel_size,batch_size,epochs,lambda_):
        self.data = data
        self.learning_rate = learning_rate
        self.filters = filters
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.lambda_ = lambda_

    def residual_block(x):
        BatchNormalization()

keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)        


    def build_model(self):
        self.regularizer = regularizers.l2(self.lambda_)
        self.model.add(Conv2D(self.filters[0],self.kernel_size,padding='same',activation='relu',kernel_regularizer = self.regularizer,input_shape=data.X_train.shape[1:]))
        self.model.add(Conv2D(self.filters[1],self.kernel_size,padding='same',activation='relu',kernel_regularizer = self.regularizer))
        self.model.add(MaxPooling2D())
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(self.filters[2],self.kernel_size,padding='same',activation='relu',kernel_regularizer = self.regularizer))
        self.model.add(Conv2D(self.filters[3],self.kernel_size,padding='same',activation='relu',kernel_regularizer = self.regularizer))
        self.model.add(MaxPooling2D())
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(512,activation = 'relu',kernel_regularizer = self.regularizer))
        self.model.add(Dropout(0.50))
        self.model.add(Dense(self.data.numclass,activation='softmax'))

    def train(self):
        self.optim = optimizers.Adam(self.learning_rate)
        self.model.compile(self.optim,'categorical_crossentropy',['accuracy'])
        datagen = ImageDataGenerator()


#        self.model.fit(self.data.X_train,self.data.Y_train,batch_size,epochs,validation_data = (self.data.X_val,self.data.Y_val),shuffle=True)

data = Data(r,numclass)
My_Model = My_Model(data,learning_rate,filters,kernel_size,batch_size,epochs,lambda_)
My_Model.build_model()
My_Model.train()
scores = My_Model.model.evaluate(data.X_test,data.Y_test,verbose=1)
print('Test loss: ', scores[0])
print('Test accuracy:',scores[1])