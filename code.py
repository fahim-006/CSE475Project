#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:34:05 2020

@author: mashrurhossainkhan
"""

#pip install tensorflow
#epochs komaiya baraiya, optimizer,loss,metrics change kore check kora jaay


#Importing libraries
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense 
import tensorflow as tf

#installing CNN
classifier = Sequential()

#step1- convolution
#relu means value ke 0/max banay

#Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(32,32,3)
classifier.add(Conv2D(32, (3, 3),activation ='relu',padding='same',input_shape = (127, 154, 3)))

#step-2 pooling
classifier.add(MaxPooling2D (pool_size=(2,2)))

#step-3 flattening
classifier.add(Flatten())

#step-4 Full connection
# 1st line er ketre around 100 newwa is a good practice and 2er power hoite hobe
classifier.add(Dense(activation ='relu',units=128))
#eita output er jonno lage...1ta output
#sigmoid karon bianry output..malaria hoiya ki hoy nai tai..sigmoid value k 0 er kachakachi nibe
#if we deal with more than 2 output(not only malaria or not, we need sufmax activationn)
classifier.add(Dense(activation ='sigmoid',units=1))

#compiling CNN, adam is an algorithm, more than 2 hoile loss='categorical_crossentrophy,'
classifier.compile(optimizer = 'adam', loss='binary_crossentropy',metrics=['accuracy' ,tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

print(classifier.summary())

#fitting CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
        ) # set validation split
    
training_set = train_datagen.flow_from_directory(
                                                'train',
                                                target_size=(127,154), #input_size
                                                batch_size=10, #number of inputs, go though the CNN
                                                class_mode='binary'  
                                               )
    
test_set = train_datagen.flow_from_directory(
                                            'test',
                                            target_size=(127,154),
                                            batch_size=32,
                                            class_mode='binary',
                                            )
#train the training set and performing in test_set
classifier.fit_generator(training_set,steps_per_epoch=int(13513/10), epochs=5,validation_data=test_set,
                           validation_steps=int(14027/10))
