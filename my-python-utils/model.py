import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras

import tensorflow as tf

def unet(pretrained_weights = None, input_size=(256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs) # 256
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1) # 128
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2) # 64
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3) # 32
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4) # 16
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    # conv10 = Conv2D(1, 1, activation = 'relu')(conv9)
    # conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv10)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = SGD(), loss = 'mse', metrics = ['accuracy'], run_eagerly=True)
    #model.compile(optimizer = SGD(lr=0.01, nesterov=True), loss = 'mse', metrics = ['accuracy'], run_eagerly=True)
    # model.compile(optimizer = Adam(lr = 1e-4), loss = 'mse', metrics = ['accuracy'])

    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


# OUTPUT_CHANNELS = 1
# def downsample(filters, size, apply_batchnorm=True):
#   initializer = tf.random_normal_initializer(0., 0.02)
#
#   result = tf.keras.Sequential()
#   result.add(
#       tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
#                              kernel_initializer=initializer, use_bias=False))
#
#   if apply_batchnorm:
#     result.add(tf.keras.layers.BatchNormalization())
#
#   result.add(tf.keras.layers.LeakyReLU())
#
#   return result
#
# def upsample(filters, size, apply_dropout=False):
#   initializer = tf.random_normal_initializer(0., 0.02)
#
#   result = tf.keras.Sequential()
#   result.add(
#     tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
#                                     padding='same',
#                                     kernel_initializer=initializer,
#                                     use_bias=False))
#
#   result.add(tf.keras.layers.BatchNormalization())
#
#   if apply_dropout:
#       result.add(tf.keras.layers.Dropout(0.5))
#
#   result.add(tf.keras.layers.ReLU())
#
#   return result
#
# def Generator():
#   down_stack = [
#     downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
#     downsample(128, 4), # (bs, 64, 64, 128)
#     downsample(256, 4), # (bs, 32, 32, 256)
#     downsample(512, 4), # (bs, 16, 16, 512)
#     downsample(512, 4), # (bs, 8, 8, 512)
#     downsample(512, 4), # (bs, 4, 4, 512)
#     downsample(512, 4), # (bs, 2, 2, 512)
#     downsample(512, 4), # (bs, 1, 1, 512)
#   ]
#
#   up_stack = [
#     upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
#     upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
#     upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
#     upsample(512, 4), # (bs, 16, 16, 1024)
#     upsample(256, 4), # (bs, 32, 32, 512)
#     upsample(128, 4), # (bs, 64, 64, 256)
#     upsample(64, 4), # (bs, 128, 128, 128)
#   ]
#
#   initializer = tf.random_normal_initializer(0., 0.02)
#
#   last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
#                                          strides=2,
#                                          padding='same',
#                                          kernel_initializer=initializer,
#                                          activation='tanh') # (bs, 256, 256, 3)
#
#   concat = tf.keras.layers.Concatenate()
#
#   inputs = tf.keras.layers.Input(shape=[256,256,1])
#   x = inputs
#
#   # Downsampling through the model
#   skips = []
#   for down in down_stack:
#     x = down(x)
#     skips.append(x)
#
#   skips = reversed(skips[:-1])
#
#   # Upsampling and establishing the skip connections
#   for up, skip in zip(up_stack, skips):
#     x = up(x)
#     x = concat([x, skip])
#
#   x = last(x)
#
#   model = tf.keras.Model(inputs=inputs, outputs=x)
#   model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
#   return model
#
#
# unet=Generator
