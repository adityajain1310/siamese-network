#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:47:32 2020

@author: aditya
"""

import os
import re
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import numpy.random as rnd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.utils as shuffle
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.regularizers  import l2
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, merge, Lambda


def w_init (shape, dtype, name = None):
    values = rnd.normal(loc = 0, scale = 1e-2, size  = shape)
    return K.variable(values, name = name, dtype = dtype)

def b_init (shape, dtype, name = None):
    values = rnd.normal(loc = 0.5, scale = 1e-2, size = shape)
    return K.variable(values, name = name, dtype = dtype)

input_shape = (98, 98, 1)
left_input = Input(input_shape)
right_input = Input(input_shape)

# build siamese network

siamnet = Sequential()
siamnet.add(Conv2D(64, (10, 10), activation = "relu", input_shape = input_shape, 
                   kernel_initializer = w_init, kernel_regularizer=l2(2e-4)))

siamnet.add(MaxPooling2D(pool_size=(2, 2)))

siamnet.add(Conv2D(128, (7, 7), activation ="relu", kernel_initializer = w_init, 
                   bias_initializer = b_init, kernel_regularizer=l2(2e-4)))

siamnet.add(MaxPooling2D(pool_size=(2, 2)))

siamnet.add(Conv2D(128, (4, 4), activation ="relu", kernel_initializer = w_init, 
                   bias_initializer = b_init, kernel_regularizer=l2(2e-4)))

siamnet.add(MaxPooling2D(pool_size=(2, 2)))

siamnet.add(Conv2D(256, (4, 4), activation ="relu", kernel_initializer = w_init,
                   bias_initializer = b_init, kernel_regularizer=l2(2e-4)))

siamnet.add(Flatten())

siamnet.add(Dense(4096, activation = "sigmoid", kernel_initializer = w_init, 
                  bias_initializer = b_init, kernel_regularizer=l2(1e-3)))

encoded_left = siamnet(left_input)
encoded_right = siamnet(right_input)

L1_siamese = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
L1_dist = L1_siamese([encoded_left, encoded_right])

similarity = Dense(1, activation = "sigmoid", bias_initializer = b_init, )(L1_dist)

siamese_network = Model(inputs = [left_input, right_input], outputs = similarity)

siamese_network.compile(loss = "binary_crossentropy", optimizer = Adam(0.00006))

siamese_network.count_params()



'''
i = cv2.imread('data/9540474/9540474.1.jpg')
g = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
k = g.reshape(98, 98, 1)
cv2.imshow('sample_img', k)
cv2.waitKey(0)
cv2.destroyAllWindows()
i.imshow()
g = g[::2, ::2]
from PIL import Image


img = Image.open('data/9540474/9540474.1.jpg').convert('LA')
k = g.reshape([98, 98, 1])
x = np.zeros([98, 98, 1])
x[:, :, :] = k


import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

img = mpimg.imread('data/9540474/9540474.1.jpg')     
gray = rgb2gray(img)    
y = gray.reshape(98, 98, 1)

plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
plt.show()'''

size = 2
sample_size = 5000  

def prepare_data(size, sample_size):
    count = 0
    path = []
    image_pairs = []
    targets =  []
    
    dim1 = 98
    dim2 = 98
    img_pair = np.zeros([sample_size, 2, dim1, dim2])
    trgt = np.zeros([sample_size, 1])
    
    for fx in os.listdir('data'):
        filepath = 'data/' + fx
        path.append(fx)
        for j in range(int(sample_size/152)):
            index1 = 0
            index2 = 0
        
            while index1 == index2:
                index1 = np.random.randint(20)
                index2 = np.random.randint(20)
                
            image1 = cv2.imread(filepath + '/' + fx + '.' + str(index1 + 1) + '.jpg')
            image2 = cv2.imread(filepath + '/' + fx + '.' + str(index2 + 1) + '.jpg')
            
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            
            gray1 = gray1[::size, ::size]
            gray2 = gray2[::size, ::size]
                        
            img_pair[count, 0, :, :] = gray1
            img_pair[count, 1, :, :] = gray2
            
            trgt[count] = 1

            count += 1
        
    count = 0

    imgImposite_pair = np.zeros([sample_size, 2, dim1, dim2])
    trgtImposite = np.zeros([sample_size, 1])
    
    for i in range(int(sample_size/20)):
        for j in range(20):
            index1 = 0
            index2 = 0
            
            while index1 == index2:
                index1 = np.random.randint(40)
                index2 = np.random.randint(40)
                
            image1 = cv2.imread('data/' + path[index1] + '/' +path[index1] + '.' + str(j+1) + '.jpg')
            image2 = cv2.imread('data/' + path[index2] + '/' +path[index2] + '.' + str(j+1) + '.jpg')
            
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            
            gray1 = gray1[::size, ::size]
            gray2 = gray2[::size, ::size]
    
            imgImposite_pair[count, 0, :, :] = gray1
            imgImposite_pair[count, 1, :, :] = gray2
            
            trgtImposite[count] = 0
            
            count += 1

    image_pairs = np.concatenate([img_pair, imgImposite_pair], axis=0)/255
    targets = np.concatenate([trgt, trgtImposite], axis=0)
    
    return image_pairs, targets

image_pairs, targets = prepare_data(size, sample_size)

no_of_sample = image_pairs.shape[0]
pairs = image_pairs.shape[1]
img_rows = image_pairs[0][0].shape[0]
img_cols = image_pairs[0][0].shape[1]

Image_pairs = image_pairs.reshape(no_of_sample, pairs, img_rows, img_cols, 1)


'''#pd.DataFrame((image_pairs)).to_csv('inputdata.csv')
#pd.DataFrame((image_pairs)).to_csv('outputdata.csv')

image_pairs = pd.read_csv('inputdata.csv')
targets = pd.read_csv('outputdata.csv')

image_pairs = image_pairs.values'''

tr_imgPairs, ts_imgPairs, tr_targets, ts_targets = train_test_split(Image_pairs, targets, 
                                                                    test_size = 0.25, 
                                                                    random_state = 0)

image1 = tr_imgPairs[:, 0]
image2 = tr_imgPairs[:, 1]

siamese_network.fit([image1, image2], tr_targets, validation_split=.25, batch_size=128, verbose=2, 
                    nb_epoch = 10)

pred = siamese_network.predict([ts_imgPairs[:, 0], ts_imgPairs[:, 1]])

def compute_accuracy(predictions, labels):
    return labels[predictions.ravel() 
                  
# print( compute_accuracy(pred, tr_targets) )