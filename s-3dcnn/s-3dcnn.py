#!/usr/bin/env python
# coding: utf-8

import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPooling3D, BatchNormalization, ZeroPadding3D
from keras.utils import np_utils

import numpy as np
from sklearn.decomposition import PCA
from data_pretreat import handle_data
from PIL import Image
import cv2

k.set_image_data_format('channels_last')
num_classes=16

# 获得数据
x_train,y_train,x_test,y_test=handle_data(train_scale=0.5,pad=1)

# 降维
def pca(x,n=196):
    pca=PCA(n_components=n)
    new_x_train=pca.fit_transform(x)
    return new_x_train

# expand_dims
x_train=np.expand_dims(x_train,axis=4)
x_test=np.expand_dims(x_test,axis=4)

#one-hot
y_train=np_utils.to_categorical(y_train)[:,1:17]
y_test =np_utils.to_categorical(y_test)[:,1:17]
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#建模
def s_3dcnn_model(input_shape,num_classes=16):
    model = Sequential()
    model.add(Conv3D(64, kernel_size=(3, 3, 11), strides=(1, 1, 1), padding='same', activation='relu',input_shape=input_shape))
    model.add(Conv3D(96, kernel_size=(1, 1, 5), strides=(1, 1, 2), padding='valid', activation='relu'))
    model.add(BatchNormalization(axis=4))
    model.add(MaxPooling3D(pool_size=(1, 1, 2)))
    model.add(Dropout(0.25))
    #model.add(ZeroPadding3D(padding=(1,1,0)))
    model.add(Conv3D(256, kernel_size=(3, 3, 5), strides=(1, 1, 2), padding='same', activation='relu'))
    model.add(Conv3D(256, kernel_size=(3, 3, 3), strides=(1, 1, 2), padding='same', activation='relu'))
    model.add(BatchNormalization(axis=4))
    model.add(MaxPooling3D(pool_size=(1, 1, 2)))
    model.add(Dropout(0.25))
    model.add(Conv3D(384, kernel_size=(3, 3, 1), strides=(1, 1, 1), padding='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    return model

print(x_train[0].shape)

s_model=s_3dcnn_model(x_train[0].shape)
s_model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  #optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

s_model.fit(x_train, y_train,
            batch_size=32,
            epochs=100,
            verbose=1)
    
print("Saving model to disk \n")
path="s_cnn_model.h5"
s_model.save(path)

print(y_test.shape)
score = s_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



