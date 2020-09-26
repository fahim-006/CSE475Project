#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 11:00:49 2020

@author: mashrurhossainkhan
"""

import numpy as np
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense 
from keras.layers import Dropout 
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
%matplotlib inline
import tensorflow as tf
import imageio
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

#installing CNN
model = Sequential()

#Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(32,32,3)
model.add(Conv2D(64, (7, 7),activation ='relu',padding='same',input_shape = (157,139,3)))

#step-2 pooling
model.add(MaxPooling2D (pool_size=(3,3), strides=(1,1)))

#step3
model.add(Conv2D(192, (3, 3),activation ='relu',padding='same',strides = (1,1)))

#step:4
model.add(MaxPooling2D (pool_size=(3,3), strides=(2,2)))

#step:5 inception(3a)
model.add(Conv2D(filters = 256, kernel_size = (1, 1), strides = (1,1), padding = 'same'))
model.add(Conv2D(filters = 256, kernel_size = (3,3), strides = (1,1), padding = 'same'))
model.add(Conv2D(filters = 256, kernel_size = (5,5), strides = (1,1), padding = 'same'))
#step:6 (adding pooling layer to inception(3a))
model.add(MaxPooling2D (pool_size=(3,3)))

#step7
model.add(Conv2D(filters = 480, kernel_size = (1, 1), strides = (1,1), padding = 'same'))
model.add(Conv2D(filters = 480, kernel_size = (3,3), strides = (1,1), padding = 'same'))
model.add(Conv2D(filters = 480, kernel_size = (5,5), strides = (1,1), padding = 'same'))
#step:8 (adding pooling layer to inception(3))
model.add(MaxPooling2D (pool_size=(3,3)))

#step9
model.add(MaxPooling2D (pool_size=(3,3), strides=(1,1)))

#step10
model.add(Conv2D(filters = 512, kernel_size = (1,1), padding = 'same'))
model.add(Conv2D(filters = 512, kernel_size = (3,3), padding = 'same'))
model.add(Conv2D(filters = 512, kernel_size = (5,5), padding = 'same'))
#(adding pooling layer to inception(3))
model.add(MaxPooling2D (pool_size=(3,3)))

#step11
model.add(Conv2D(filters = 512, kernel_size = (1, 1), strides = (1,1), padding = 'same'))
model.add(Conv2D(filters = 512, kernel_size = (3,3), strides = (1,1), padding = 'same'))
model.add(Conv2D(filters = 512, kernel_size = (5,5), strides = (1,1), padding = 'same'))
#(adding pooling layer to inception(3))
#model.add(MaxPooling2D (pool_size=(2,2)))

#step12
model.add(Conv2D(filters = 512, kernel_size = (1, 1), strides = (1,1), padding = 'same'))
model.add(Conv2D(filters = 512, kernel_size = (3,3), strides = (1,1), padding = 'same'))
model.add(Conv2D(filters = 512, kernel_size = (5,5), strides = (1,1), padding = 'same'))
#(adding pooling layer to inception(3))
#model.add(MaxPooling2D (pool_size=(100,100)))

#step13
model.add(Conv2D(filters = 528, kernel_size = (1, 1), strides = (1,1), padding = 'same'))
model.add(Conv2D(filters = 528, kernel_size = (3,3), strides = (1,1), padding = 'same'))
model.add(Conv2D(filters = 528, kernel_size = (5,5), strides = (1,1), padding = 'same'))
#(adding pooling layer to inception(3))
#model.add(MaxPooling2D (pool_size=(3,3)))

#step14
model.add(Conv2D(filters = 832, kernel_size = (1, 1), strides = (1,1), padding = 'same'))
model.add(Conv2D(filters = 832, kernel_size = (3,3), strides = (1,1), padding = 'same'))
model.add(Conv2D(filters = 832, kernel_size = (5,5), strides = (1,1), padding = 'same'))
#(adding pooling layer to inception(3))
#model.add(MaxPooling2D (pool_size=(3,3)))

#step15
#model.add(MaxPooling2D (pool_size=(5,5), strides=(2,2)))

#step16
model.add(Conv2D(filters = 832, kernel_size = (1, 1), strides = (1,1), padding = 'same'))
model.add(Conv2D(filters = 832, kernel_size = (3,3), strides = (1,1), padding = 'same'))
model.add(Conv2D(filters = 832, kernel_size = (5,5), strides = (1,1), padding = 'same'))
#(adding pooling layer to inception(3))
#model.add(MaxPooling2D (pool_size=(3,3)))

#step17
model.add(Conv2D(filters = 1024, kernel_size = (1, 1), strides = (1,1), padding = 'same'))
model.add(Conv2D(filters = 1024, kernel_size = (3,3), strides = (1,1), padding = 'same'))
model.add(Conv2D(filters = 1024, kernel_size = (5,5), strides = (1,1), padding = 'same'))
#(adding pooling layer to inception(3))
#model.add(MaxPooling2D (pool_size=(3,3)))

#step18
model.add(AveragePooling2D(pool_size=(1,1), strides =(1,1)))

#step19
model.add(Dropout(0.4))

#step20
model.add(Flatten())

#step21
model.add(Dense(activation = 'linear', units=1))

#step22
classes=2
model.add(Dense(classes, activation='softmax', name='fc' + str(classes)))

#step23
model.compile(optimizer = 'adam', loss='binary_crossentropy',  metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

    
from keras.preprocessing.image import ImageDataGenerator
        
train_datagen = ImageDataGenerator(
            rescale=1./255, #pixel value will be within 0 and 1
            shear_range=0.2, #Shear angle in counter-clockwise direction in degrees
            zoom_range=0.2,
            horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                           'train/',
                           target_size=(157,139), #input_size
                           batch_size=32, #number of inputs, go though the CNN
                           class_mode='categorical')

test_set = test_datagen.flow_from_directory(
            'test/',
            target_size=(157,139),
            batch_size=32,
            class_mode='categorical')

model.summary()

#train the training set and performing in test_set
model.fit(training_set,steps_per_epoch=int(13513/10),batch_size=200, epochs=2,validation_data=test_set,
                           validation_steps=int(14027/10))

    
    img_path = 'images/my_image.png'
    img = image.load_img(img_path, target_size=(157,139))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)
    my_image = imageio.imread(img_path)
    imshow(my_image)
    print("class prediction vector [p(0), p(1) = ")

    
    z= (model.predict((x)))
    y=(max(z[0][1],z[0][0]))
    print(z)
    print(y)
    if(z[0][0]==y):
        print("MALARIA NEGATIVE")
    else:
        print("MALARIA POSITIVE")
        
        
    model.summary()
    
    plot_model(model, to_file='model.png')
    SVG(model_to_dot(model).create(prog='dot', format='svg'))


