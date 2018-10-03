# DongKyu Kim
# ECE 471 CGML Assignment 4
# CIFAR-10
# Professor Curro

# library imports
import numpy as np
from tqdm import tqdm
import keras
import tensorflow as tf
from keras import optimizers, regularizers
from keras.datasets import cifar100
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, Input


# Parameters
r = 0.9 # Ratio of training data otu of training + validation
batch_size = 64
epochs = 20
numclass = 100

# hyper parameters
learning_rate = 0.004

# Useful functions
def one_hot_encoding(data,numclass):
    targets = np.array(data).reshape(-1)
    return np.eye(numclass)[targets]


# Data Generation
class Data(object):
    def __init__(self,r,numclass): # N is number of training data that will be used
        (self.xtemp,self.ytemp),(self.X_test,self.Y_test) = cifar100.load_data()
        mask = np.full(self.xtemp.shape[0],True)
        mask[np.random.choice(self.xtemp.shape[0], int(self.xtemp.shape[0]*(1-r)), replace=False)] = False
        self.numclass = numclass
        self.X_train = self.xtemp[mask].astype('float32')
        self.Y_train = self.ytemp[mask]
        self.X_val = self.xtemp[~mask].astype('float32')
        self.Y_val = self.ytemp[~mask]
        self.X_test = self.X_test.astype('float32')
        self.Y_test = self.Y_test
        self.X_train = self.normalize(self.X_train)
        self.X_val = self.normalize(self.X_val)
        self.X_test = self.normalize(self.X_test)
        self.Y_train = one_hot_encoding(self.Y_train,self.numclass)
        self.Y_val = one_hot_encoding(self.Y_val,self.numclass)
        self.Y_test = one_hot_encoding(self.Y_test,self.numclass)
    def normalize(self,X):
        X /= 255
        mu = [0.4914,0.4822,0.4465] # These values are from https://github.com/kuangliu/pytorch-cifar/issues/19
        std = [0.2023,0.1994,0.2010] 
        for i in range(0,2):
            X[:,:,:,i] = (X[:,:,:,i]-mu[i])/std[i]
        return X

# Model
class My_Model(object):
    def __init__(self,data,learning_rate,batch_size,epochs):
        self.data = data
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def convconv(self,filters,kernel_size,strides):
        return Conv2D(filters,kernel_size,strides =strides,padding='same',activation='relu')

    def residual_block(self,input,filter): #This is from https://arxiv.org/pdf/1603.05027.pdf Identity Mappings in Deep Residual Networks
        x = BatchNormalization()(input)
        x = Activation('relu')(x)
        x = self.convconv(filter,[3,3],1)(x)
        x = BatchNormalization()(input)
        x = Activation('relu')(x)
        x = self.convconv(filter,[3,3],1)(x)
        return keras.layers.add([input,x])

    def build_model(self):
        input = Input(shape = self.data.X_train.shape[1:])
        x = self.convconv(64,[7,7],2)(input)
        x = MaxPooling2D(3,strides = 2)(x)
        for i in range(0,2):
            x = self.residual_block(x,64)
        x = self.convconv(128,[3,3],2)(x)
        for i in range(0,2):
            x = self.residual_block(x,128)
        x = self.convconv(256,[3,3],2)(x)
        for i in range(0,4):
            x = self.residual_block(x,256)        
        x = self.convconv(512,[3,3],2)(x)
        for i in range(0,2):
            x = self.residual_block(x,512)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = Dense(1024,activation = 'relu')(x)
        y = Dense(self.data.numclass,activation='softmax')(x)
        self.model = Model(inputs = input, outputs = y)

    def train(self):
        self.optim = optimizers.SGD(self.learning_rate,momentum=0.9,decay=5e-4)
        self.model.compile(self.optim,'categorical_crossentropy',['accuracy','top_k_categorical_accuracy'])
        self.datagen = keras.preprocessing.image.ImageDataGenerator(
            horizontal_flip = True,fill_mode = 'constant',
            width_shift_range = 4, height_shift_range = 4
            )
        self.datagen.fit(self.data.X_train)
        self.model.fit_generator(self.datagen.flow(self.data.X_train, self.data.Y_train,batch_size=self.batch_size),
            steps_per_epoch=len(self.data.X_train)/self.batch_size,epochs=self.epochs,validation_data = (self.data.X_val,self.data.Y_val),verbose=1)

        #self.model.fit(self.data.X_train,self.data.Y_train,self.batch_size,self.epochs,validation_data = (self.data.X_val,self.data.Y_val),shuffle=True)

data = Data(r,numclass)
My_Model = My_Model(data,learning_rate,batch_size,epochs)
My_Model.build_model()
My_Model.train()
scores = My_Model.model.evaluate(data.X_test,data.Y_test,verbose=2)
print('Test loss: ', scores[0])
print('Test top 1 accuracy:',scores[1])
print('Test top 5 accuracy:',scores[2])

# I stopped and checked this code on cifar-100 when cifar-10 reached 0.8042 validation accuracy. The top5 accuracy for validation is 0.7435 and
# test accuracy is 0.7465. This is over the minimum requirement. The model was run for 20 epochs. It looks like if I did more epochs I will get a 
# better accuracy, but instead of running it, I decided to make a better model.
