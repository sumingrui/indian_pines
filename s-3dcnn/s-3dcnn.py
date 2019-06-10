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
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau

import os
import numpy as np
from sklearn.decomposition import PCA
from data_pretreat import handle_data
from PIL import Image
import cv2
import traceback
from functools import partial

# 获得模型名称
def get_model_name():
    return traceback.extract_stack()[-2][2]

# 降维
def pca(x,n=196):
    pca=PCA(n_components=n)
    new_x_train=pca.fit_transform(x)
    return new_x_train

# 创建子文件夹
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

# 3*3 模型
def s3_1_3dcnn_model(input_shape,num_classes=16):
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
    savedir = get_model_name()

    return model, savedir


# 3*3 模型
def s3_2_3dcnn_model(input_shape,num_classes=16):
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
    x = Dense(num_classes, activation='softmax', name='fc1')(x)

    model = Model(inputs = x_input, outputs = x, name = get_model_name())
    model.summary()
    savedir = get_model_name()

    return model, savedir


# 3*3 模型 简化滤波器数量
def s3_3_3dcnn_model(input_shape,num_classes=16):
    x_input = Input(input_shape)
    x = Conv3D(16, kernel_size=(3, 3, 11), strides=(1, 1, 1), padding='same', name = 'conv1')(x_input)
    x = BatchNormalization(axis=4, name = 'bn1')(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(1, 1, 2))(x)

    x = Dropout(0.25)(x)
    x = Conv3D(32, kernel_size=(3, 3, 5), strides=(1, 1, 1), padding='same', name = 'conv2')(x)
    x = BatchNormalization(axis=4, name = 'bn2')(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(1, 1, 2))(x)

    x = Dropout(0.25)(x)
    x = Conv3D(64, kernel_size=(2, 2, 5), strides=(1, 1, 2), padding='valid', name = 'conv3')(x)
    x = BatchNormalization(axis=4, name = 'bn3')(x)
    x = Activation('relu')(x)

    x = AveragePooling3D(pool_size=(2,2,2), name = 'avg_pool')(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax', name='fc1')(x)

    model = Model(inputs = x_input, outputs = x, name = get_model_name())
    model.summary()
    savedir = get_model_name()

    return model, savedir


# 7*7 3dcnn模型
def s7_1_3dcnn_model(input_shape,num_classes=16):
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
    x = Dense(num_classes, activation='softmax', name='fc1')(x)

    model = Model(inputs = x_input, outputs = x, name = get_model_name())
    model.summary()
    savedir = get_model_name()

    return model, savedir

# 7*7 3dcnn模型 简化滤波器数量
def s7_2_3dcnn_model(input_shape,num_classes=16):
    x_input = Input(input_shape)
    x = Conv3D(16, kernel_size=(1, 1, 11), strides=(1, 1, 1), padding='valid', name = 'conv1', kernel_regularizer=l2(0.01))(x_input)
    x = BatchNormalization(axis=4, name = 'bn1')(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(1, 1, 2))(x)

    x = Dropout(0.25)(x)
    x = Conv3D(32, kernel_size=(3, 3, 7), strides=(1, 1, 1), padding='valid', name = 'conv2', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization(axis=4, name = 'bn2')(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(1, 1, 2))(x)

    x = Dropout(0.25)(x)
    x = Conv3D(128, kernel_size=(3, 3, 5), strides=(1, 1, 1), padding='valid', name = 'conv3', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization(axis=4, name = 'bn3')(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(1, 1, 2))(x)

    x = Dropout(0.25)(x)
    x = Conv3D(256, kernel_size=(2, 2, 5), strides=(1, 1, 1), padding='valid', name = 'conv4', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization(axis=4, name = 'bn4')(x)
    x = Activation('relu')(x)

    x = AveragePooling3D(pool_size=(2,2,2), name = 'avg_pool')(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax', name='fc1')(x)

    model = Model(inputs = x_input, outputs = x, name = get_model_name())
    model.summary()
    savedir = get_model_name()

    return model, savedir


# 5*5 input
def s5_1_3dcnn_model(input_shape,num_classes=16):
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
    x = Conv3D(128, kernel_size=(2, 2, 5), strides=(1, 1, 1), padding='valid', name = 'conv3')(x)
    x = BatchNormalization(axis=4, name = 'bn3')(x)
    x = Activation('relu')(x)

    x = AveragePooling3D(pool_size=(2,2,2), name = 'avg_pool')(x)
    x = Flatten()(x)
    # 增加一层FC
    # x = Dropout(0.25)(x)
    # x = Dense(400, activation='tanh', kernel_regularizer=l2(0.01), name = 'fc1')(x)
    # x = BatchNormalization(axis=1, name = 'bn4')(x)
    x = Dense(num_classes, activation='softmax', name='fc1')(x)

    model = Model(inputs = x_input, outputs = x, name = get_model_name())
    model.summary()
    savedir = get_model_name()
    # mkdir(savedir)
    # plot_model(model, to_file=savedir+'/'+savedir+'.png', show_shapes=True)
    return model, savedir


# 设置学习率
def lr_schedule(epoch,lr_init,lr_by_epoch,lr_scale):
    # lr = 1e-3
    # base = 1/3
    # if epoch >0:
    #     t = int(epoch / 30)
    #     lr = lr * base**t
    # print('Learning rate: ', lr)

    # return lr

    lr = lr_init
    base = lr_scale
    if epoch >0:
        t = int(epoch / lr_by_epoch)
        lr = lr * base**t
    print('Learning rate: ', lr)

    return lr


def run_model(model, model_name, lr_init, lr_by_epoch, lr_scale, n_batch_size, n_epoch, k_optimizer):
    # ReduceLROnPlateau和LearningRateScheduler选一个
    lr_s = partial(lr_schedule,lr_init=lr_init, lr_by_epoch=lr_by_epoch, lr_scale=lr_scale)
    lr_scheduler = LearningRateScheduler(lr_s)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=lr_scale, patience=5, mode='auto')

    tensorboard = TensorBoard(log_dir=model_name+'/logs')

    checkpoint = ModelCheckpoint(filepath=model_name+'/bset_acc.h5',
                                    monitor='val_acc',
                                    verbose=1,
                                    save_best_only='True',
                                    mode='max',
                                    period=1)

    callback_lists = [tensorboard, checkpoint, reduce_lr]

    if k_optimizer=='adam':
        model.compile(loss=keras.losses.categorical_crossentropy,
                        #optimizer=keras.optimizers.Adadelta(),
                        optimizer=keras.optimizers.Adam(lr=lr_init),
                        metrics=['accuracy','crossentropy'])

    model.fit(x_train, y_train,
                batch_size=n_batch_size,
                epochs=n_epoch,
                # callbacks=[lr_scheduler, tensorboard, checkpoint],
                callbacks=callback_lists,
                validation_data=(x_test, y_test),
                verbose=1)
        
    print("Saving model to disk \n")
    path=model_name+'/'+model_name+'.h5'
    model.save(path)

    print(y_test.shape)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])



if __name__ == '__main__':
    K.set_image_data_format('channels_last')
    
    # 设置参数
    train_scale=0.3     # 训练集比例
    kernel_size=7       # 样本尺寸
    bf=True            # 是否滤波

    # 如果用ReduceLROnPlateau就不需要lr_by_epoch参数
    lr_init=0.01
    lr_by_epoch=30
    lr_scale=1/3
    n_batch_size=32
    n_epoch=300
    optimizer='adam'

    
    # 获得数据
    x_train,y_train,x_test,y_test=handle_data(train_scale,kernel_size,bf)

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

    s_model, model_name = s7_2_3dcnn_model(x_train[0].shape)

    mkdir(model_name)
    plot_model(s_model, to_file=model_name+'/'+model_name+'.png', show_shapes=True)

    run_model(s_model, model_name, lr_init, lr_by_epoch, lr_scale, n_batch_size, n_epoch, optimizer)