# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 20:44:56 2018

@author: Onkar
"""

import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from sklearn.preprocessing import normalize
from keras import optimizers

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import h5py




def load_dataset():
    '''
    Loads the training and the test data
    '''
    
    train_dataset = h5py.File('train_happy.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('test_happy.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def model(input_shape):
    '''
    Defines the model
    
    Arguments
    input_shape  = shape of the input image
    
    Output
    A model instance in keras
    '''
    #Defines the input place holder with shape input_shape
    X_in = Input(input_shape)
    
    #pads the border of the input image with zeros
    x = ZeroPadding2D((3,3))(X_in)
    
    #Convolutional layer 1
    x = Conv2D(32, (7,7), strides = (1,1), name = "Conv1")(x)
    
    #Batch Normalization
    x = BatchNormalization(axis = 3, name = "BatchNorm1")(x)
    
    #Non-Linear Activation
    x = Activation('relu')(x)
    
    # MAXPOOL
    x = MaxPooling2D((2, 2), name='max_pool1')(x)
    
    #-----------------------------------------------
    
    #Convolutional layer 2
    #x = Conv2D(64, (5,5), strides = (1,1), name = "Conv2")(x)
    
    #Batch Normalization
    #x = BatchNormalization(axis = 3, name = "BatchNorm2")(x)
    
    #Non-Linear Activation
    #x = Activation('relu')(x)
    
    # MAXPOOL
    #x = MaxPooling2D((2, 2), name='max_pool2')(x)
    
    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    x = Flatten()(x)
    x = Dense(100, activation='relu', name='fc1')(x)
    x = Dense(50, activation='relu', name='fc2')(x)
    x = Dense(1, activation = 'sigmoid', name = 'fc3' )(x)
    
    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_in, outputs = x, name='HappyFace')
    
    return model

#Getting the dataset
Xtrain_orig, Ytrain_orig, Xtest_orig, Ytest_orig, classes = load_dataset()

#normalizing the data
Xtrain = Xtrain_orig/255
Xtest = Xtest_orig/255

#Reshaping the labels
Ytrain = Ytrain_orig.T
Ytest = Ytest_orig.T    

#Getting the model
model = model(Xtrain[0,:,:,:].shape)

#Compiling the model
optimizer = optimizers.SGD(lr = 0.01)
model.compile(optimizer = optimizer , loss = 'binary_crossentropy', metrics = ['accuracy'])

#fitting the model
model.fit(x = Xtrain, y=Ytrain, batch_size = 32, epochs = 40)

Loss, accuracy = model.evaluate(x = Xtest, y = Ytest, batch_size = 32)

print("The Accuracy of the network is {} %".format(accuracy*100))

imshow(Xtest[1,:,:,:])