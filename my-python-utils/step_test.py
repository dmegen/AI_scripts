#!/usr/bin/env python

import tensorflow as tf
import xarray as xr
import model

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler


def simple_model(pretrained_weights = None, input_size=(256,256,1)):
    inputs = tf.keras.Input(input_size)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs) # 256
    conv1 = tf.keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = tf.keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = tf.keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = tf.keras.layers.Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = tf.keras.layers.Conv2D(1, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)

    model = tf.keras.models.Model(inputs = inputs, outputs = conv1)

    model.compile(optimizer = tf.keras.optimizers.Adam(lr = 1e-4), loss = "mean_absolute_error", metrics = ['accuracy'], run_eagerly=True)

    model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def full_model(pretrained_weights = None, input_size=(256,256,1)):

    inputs = tf.keras.Input(input_size)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs) # 256
    conv1 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
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

    model = tf.keras.models.Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = tf.keras.optimizers.Adam(lr = 1e-4), loss = "mean_absolute_error", metrics = ['accuracy'], run_eagerly=True)

    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

def load_data():
    ntimes = 1000
    data = xr.open_dataset("wrf_data/CTRLradrefl/REFL/wrf2d_d01_CTRL_REFL_10CM_201007-201009.nc")
    x = data.REFL_10CM[:ntimes*3:3,350:350+256,650:650+256].values

    ds = xr.open_mfdataset("wrf_data/CTRL3D/2010/wrf3d_d01_CTRL_W_20100[789]*")
    y = ds.W[:ntimes,20,350:350+256,650:650+256].values

    return x,y


def medium_data(level=20, method="sel"):
    data = xr.open_mfdataset("wrf_data/CTRLradrefl/REFL/wrf2d_d01_CTRL_REFL_10CM_201*07-201*09.nc")
    x = data.REFL_10CM[::3,350:350+256,650:650+256].values

    # try:
    #     ds = xr.opendataset("preloaded/wrf_conus_Jul-Sept_W.nc")
    # except:
    ds = xr.open_mfdataset("wrf_data/CTRL3D/20*/wrf3d_d01_CTRL_W_201?0[789]*.nc")
    if method=="sel":
        y = ds.W[:,level,350:350+256,650:650+256].values
    elif method=="max":
        y = ds.W[:,:,350:350+256,650:650+256].values.max(axis=1)
    elif method=="mean":
        y = ds.W[:,:,350:350+256,650:650+256].values.mean(axis=1)


    return x,y



def large_data(level=20):
    data = xr.open_mfdataset("wrf_data/CTRLradrefl/REFL/wrf2d_d01_CTRL_REFL_10CM_20*07-20*09.nc")
    x = data.REFL_10CM[::3,350:350+256,650:650+256].values

    try:
        ds = xr.opendataset("preloaded/wrf_conus_Jul-Sept_W.nc")
    except:
        ds = xr.open_mfdataset("wrf_data/CTRL3D/20*/wrf3d_d01_CTRL_W_20??0[789]*.nc")
    y = ds.W[:,level,350:350+256,650:650+256].values

    return x,y

def main():
    tf.test.is_gpu_available()

    x,y = load_data()
    # train_dataset = tf.data.Dataset.from_tensor_slices((x, y))

    unet = simple_model(input_size=(256,256, 1))
    # unet = model.unet(input_size=(256,256, 1))

    train_dataset = tf.data.Dataset.from_tensor_slices((x[:,:,:,np.newaxis], y[:,:,:,np.newaxis]))
    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 100
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

    unet.fit(train_dataset, epochs=5)

if __name__ == '__main__':
    main()
