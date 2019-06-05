#!/usr/bin/env python
# coding: utf-8

import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Conv3D, MaxPooling3D, AveragePooling3D, BatchNormalization, ZeroPadding3D
from keras.regularizers import l2
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler
from keras.utils import plot_model

import numpy as np
from sklearn.decomposition import PCA
from data_pretreat import handle_data
from PIL import Image
import cv2


# 降维
def pca(x,n=196):
    pca=PCA(n_components=n)
    new_x_train=pca.fit_transform(x)
    return new_x_train


# 第一种3dcnn模型
def s1_3dcnn_model(input_shape,num_classes=16):
    model = Sequential()
    model.add(Conv3D(64, kernel_size=(3, 3, 11), strides=(1, 1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01), input_shape=input_shape))
    model.add(Conv3D(96, kernel_size=(1, 1, 5), strides=(1, 1, 2), padding='valid', activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization(axis=4))
    model.add(MaxPooling3D(pool_size=(1, 1, 2)))

    model.add(Dropout(0.25))
    #model.add(ZeroPadding3D(padding=(1,1,0)))
    model.add(Conv3D(256, kernel_size=(3, 3, 5), strides=(1, 1, 2), padding='same', activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Conv3D(256, kernel_size=(3, 3, 3), strides=(1, 1, 2), padding='same', activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization(axis=4))
    model.add(MaxPooling3D(pool_size=(1, 1, 2)))

    model.add(Dropout(0.25))
    model.add(Conv3D(384, kernel_size=(3, 3, 1), strides=(1, 1, 1), padding='valid', activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization(axis=4))
    model.add(Flatten())

    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()
    plot_model(model, to_file='s1_3dcnn.png', show_shapes=True)
    return model


# 第二种3dcnn模型
def s2_3dcnn_model(input_shape,num_classes=16):
    x_input = Input(input_shape)
    x = Conv3D(64, kernel_size=(1, 1, 11), strides=(1, 1, 1), padding='same', name = 'conv1')(x_input)
    x = BatchNormalization(axis=4, name = 'bn1')(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(1, 1, 2))(x)

    x = Dropout(0.25)(x)
    x = Conv3D(256, kernel_size=(2, 2, 5), strides=(1, 1, 2), padding='valid', name = 'conv2')(x)
    x = BatchNormalization(axis=4, name = 'bn2')(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(1, 1, 2))(x)

    x = Dropout(0.25)(x)
    x = Conv3D(1024, kernel_size=(1, 1, 5), strides=(1, 1, 2), padding='valid', name = 'conv3')(x)
    x = BatchNormalization(axis=4, name = 'bn3')(x)
    x = Activation('relu')(x)

    x = AveragePooling3D(pool_size=(2,2,2), name = 'avg_pool')(x)
    x = Flatten()(x)
    # 增加一层FC
    # x = Dropout(0.25)(x)
    # x = Dense(400, activation='tanh', kernel_regularizer=l2(0.01), name = 'fc1')(x)
    # x = BatchNormalization(axis=1, name = 'bn4')(x)
    x = Dense(num_classes, activation='softmax', name='fc1')(x)

    model = Model(inputs = x_input, outputs = x, name = 's2_model')
    model.summary()
    plot_model(model, to_file='s2_3dcnn.png', show_shapes=True)
    return model


# 第三种3dcnn模型
def s3_3dcnn_model(input_shape,num_classes=16):
    x_input = Input(input_shape)
    x = Conv3D(64, kernel_size=(3, 3, 11), strides=(1, 1, 1), padding='same', name = 'conv1')(x_input)
    x = BatchNormalization(axis=4, name = 'bn1')(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(1, 1, 2))(x)

    x = Dropout(0.25)(x)
    x = Conv3D(256, kernel_size=(3, 3, 5), strides=(1, 1, 2), padding='same', name = 'conv2')(x)
    x = BatchNormalization(axis=4, name = 'bn2')(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(1, 1, 2))(x)

    x = Dropout(0.25)(x)
    x = Conv3D(1024, kernel_size=(2, 2, 5), strides=(1, 1, 2), padding='valid', name = 'conv3')(x)
    x = BatchNormalization(axis=4, name = 'bn3')(x)
    x = Activation('relu')(x)

    x = AveragePooling3D(pool_size=(2,2,2), name = 'avg_pool')(x)
    x = Flatten()(x)
    # 增加一层FC
    # x = Dropout(0.25)(x)
    # x = Dense(400, activation='tanh', kernel_regularizer=l2(0.01), name = 'fc1')(x)
    # x = BatchNormalization(axis=1, name = 'bn4')(x)
    x = Dense(num_classes, activation='softmax', name='fc1')(x)

    model = Model(inputs = x_input, outputs = x, name = 's3_model')
    model.summary()
    plot_model(model, to_file='s3_3dcnn.png', show_shapes=True)
    return model


# 7*7 3dcnn模型
def s4_3dcnn_model(input_shape,num_classes=16):
    x_input = Input(input_shape)
    x = Conv3D(64, kernel_size=(3, 3, 11), strides=(1, 1, 1), padding='valid', name = 'conv1')(x_input)
    x = BatchNormalization(axis=4, name = 'bn1')(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(1, 1, 2))(x)

    x = Dropout(0.25)(x)
    x = Conv3D(256, kernel_size=(3, 3, 7), strides=(1, 1, 1), padding='valid', name = 'conv2')(x)
    x = BatchNormalization(axis=4, name = 'bn2')(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(1, 1, 2))(x)

    x = Dropout(0.25)(x)
    x = Conv3D(1024, kernel_size=(2, 2, 5), strides=(1, 1, 1), padding='valid', name = 'conv3')(x)
    x = BatchNormalization(axis=4, name = 'bn3')(x)
    x = Activation('relu')(x)

    x = AveragePooling3D(pool_size=(2,2,2), name = 'avg_pool')(x)
    x = Flatten()(x)
    # 增加一层FC
    # x = Dropout(0.25)(x)
    # x = Dense(400, activation='tanh', kernel_regularizer=l2(0.01), name = 'fc1')(x)
    # x = BatchNormalization(axis=1, name = 'bn4')(x)
    x = Dense(num_classes, activation='softmax', name='fc1')(x)

    model = Model(inputs = x_input, outputs = x, name = 's4_model')
    model.summary()
    plot_model(model, to_file='s4_3dcnn.png', show_shapes=True)
    return model

# 7*7 3dcnn模型
def s5_3dcnn_model(input_shape,num_classes=16):
    x_input = Input(input_shape)
    x = Conv3D(16, kernel_size=(1, 1, 11), strides=(1, 1, 1), padding='valid', name = 'conv1')(x_input)
    x = BatchNormalization(axis=4, name = 'bn1')(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(1, 1, 2))(x)

    x = Dropout(0.25)(x)
    x = Conv3D(32, kernel_size=(3, 3, 7), strides=(1, 1, 1), padding='valid', name = 'conv2')(x)
    x = BatchNormalization(axis=4, name = 'bn2')(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(1, 1, 2))(x)

    x = Dropout(0.25)(x)
    x = Conv3D(128, kernel_size=(3, 3, 5), strides=(1, 1, 1), padding='valid', name = 'conv3')(x)
    x = BatchNormalization(axis=4, name = 'bn3')(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(1, 1, 2))(x)

    x = Dropout(0.25)(x)
    x = Conv3D(256, kernel_size=(2, 2, 5), strides=(1, 1, 1), padding='valid', name = 'conv4')(x)
    x = BatchNormalization(axis=4, name = 'bn4')(x)
    x = Activation('relu')(x)

    x = AveragePooling3D(pool_size=(2,2,2), name = 'avg_pool')(x)
    x = Flatten()(x)
    # 增加一层FC
    # x = Dropout(0.25)(x)
    # x = Dense(400, activation='tanh', kernel_regularizer=l2(0.01), name = 'fc1')(x)
    # x = BatchNormalization(axis=1, name = 'bn4')(x)
    x = Dense(num_classes, activation='softmax', name='fc1')(x)

    model = Model(inputs = x_input, outputs = x, name = 's5_model')
    model.summary()
    plot_model(model, to_file='s5_3dcnn.png', show_shapes=True)
    return model


def run_model(model):
    lr_scheduler = LearningRateScheduler(lr_schedule)
    model.compile(loss=keras.losses.categorical_crossentropy,
                    #optimizer=keras.optimizers.Adadelta(),
                    optimizer=keras.optimizers.Adam(lr=lr_schedule(0)),
                    metrics=['accuracy'])

    model.fit(x_train, y_train,
                batch_size=32,
                epochs=300,
                callbacks=[lr_scheduler],
                verbose=1)
        
    print("Saving model to disk \n")
    path="s_cnn_model.h5"
    model.save(path)

    print(y_test.shape)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


# 设置学习率
def lr_schedule(epoch):
    # 每隔100个epoch，学习率减小为原来的1/3
    # if epoch % 100 == 0 and epoch != 0:
    #     lr = K.get_value(s_model.optimizer.lr)
    #     K.set_value(s_model.optimizer.lr, lr * 0.33)
    #     print("lr changed to {}".format(lr * 0.33))
    # return K.get_value(s_model.optimizer.lr)

    # lr = 1e-3
    # if epoch > 200:
    #     lr *= 1/81
    # elif epoch > 150:
    #     lr *= 1/27
    # elif epoch > 100:
    #     lr *= 1/9
    # elif epoch > 50:
    #     lr *= 1/3
    # print('Learning rate: ', lr)

    lr = 1e-3
    base = 1/4
    if epoch >0:
        t = int(epoch / 50)
        lr = lr * base**t
    print('Learning rate: ', lr)

    return lr



if __name__ == '__main__':
    K.set_image_data_format('channels_last')
    num_classes=16

    # 获得数据
    x_train,y_train,x_test,y_test=handle_data(train_scale=0.3,kernel_size=7)

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

    s_model = s5_3dcnn_model(x_train[0].shape)
    run_model(s_model)