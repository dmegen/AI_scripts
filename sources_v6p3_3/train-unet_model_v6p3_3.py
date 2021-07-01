#!/usr/bin/env python

import neptune
import neptune_tensorboard as neptune_tb
from neptunecontrib.monitoring.keras import NeptuneMonitor

import optuna

import keras.backend as K
import tensorflow as tf
import xarray as xr
import numpy as np
import os
import sys
from scipy.ndimage import gaussian_filter
from tensorflow.keras.optimizers import *
import glob
import time

t = time.strftime("%Y_%m_%d_%H_%M", time.localtime())

def scheduler(epoch):
    if epoch > 0:
        return 0.01 / epoch
    else:
        return 0.01

def custom_loss(y_true, y_pred, thresh):
    y_true2 = y_true[y_true > thresh + 0.1]
    y_pred2 = y_pred[y_true > thresh + 0.1]
    mse = tf.keras.losses.MeanSquaredError()
    huber = tf.keras.losses.Huber()
    return mse(y_true2, y_pred2)

def cust_loss(thresh):
    def loss(y_true, y_pred):
        return custom_loss(y_true, y_pred, thresh)
    return loss

#
# Mean Absolute Error metric
#
def custom_mae(y_true, y_pred, thresh):
    y_true2 = y_true[y_true > thresh + 0.1]
    y_pred2 = y_pred[y_true > thresh + 0.1]
    mse = tf.keras.losses.MeanSquaredError()
    return mse(y_true2, y_pred2)
def cust_mae(thresh):
    def cmae(y_true, y_pred):
        return custom_mae(y_true, y_pred, thresh)
    return cmae

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    coef = (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
    return coef

def custom_accuracy(y_true, y_pred , thresh):
    y_true_t = y_true[y_true > thresh + 0.1]
    y_pred_t = y_pred[y_true > thresh + 0.1]
    eval = K.mean(K.less(K.abs(y_true_t - y_pred_t), 0.05))
    return eval
def cust_accuracy(thresh):
    def cacc(y_true, y_pred):
        return custom_accuracy(y_true, y_pred, thresh)
    return cacc

############################
# set up the run information
############################

feature_data          = '/glade/work/hardt/ds612/model2_15minuteAfterHour_3D_refl_shuffled_scaled-v6.nc'
label_data            = '/glade/work/hardt/ds612/model2_composite_W_shuffled_scaled-v6.nc'

from unet_model_v6p3_3 import unet

#
# Set up neptune
#
NEPTUNE_API_TOKEN = os.environ.get('NEPTUNE_API_TOKEN')
neptune.init(project_qualified_name='hardt/Pred-W-RefOffset',
             api_token=NEPTUNE_API_TOKEN)

neptune_experiment_name = 'v6p3_3 ref15minAfter Wcomp thresh reflComp_-35.0'
neptune_upload_source_files = ['train-unet_model_v6p3_3.py', 'unet_model_v6p3_3.py']
neptune_tags = ['v6p3_3', '15minOffset', 'Shuffled','CustomLoss', 'Adam', 'unet']

output_root_directory = '/glade/work/hardt/models'

#
# For lable_feature_threshold
#
# Based on the min and max from scaling the data
# V = Vs * (refl_scaling_per99.99 - refl_scaling_min) - refl_scaling_min
#   = Vs * 93.3865 + -35
# a value of 0.001 would be -34.9066
# a value of 0.4 would be ~02dbz
# a value of 0.5 would be ~11dbz
# a value of 0.6 would be ~20dbz
# a value of 0.7 would be ~30dbz
# 

PARAMS = {'epochs': 100,
          'batch_size': 32,
          'optimizer': 'Adam',
          'learning_rate': 0.0001,
          'beta_1': 0.9,
          'beta_2': 0.999,
          'epsilon': 1.0,
          'decay': 0.0,
          'momentum': 0.9,
          'custLossThresh': -99.0,
          'label_level': 'composite',
          'label_feature_threshold': 0.001,
          'refl_scaling_min': -35.0,
          'refl_scaling_per99.99': 45.6660232543945,
          'W_scaling_min': -14.29787,
          'W_scaling_per99.99': 0.288602113723755,
          'model_run_name': 'unet_v6p3',
          'feature_description': '15minAfterHour_refl',
          }

neptune.create_experiment(name=neptune_experiment_name, 
                          params=PARAMS,
                          upload_source_files=neptune_upload_source_files,
                          tags=neptune_tags
)

#--------------------------
#
if PARAMS['optimizer'] == 'Adam':
  optimizer = Adam(lr=PARAMS['learning_rate'],
                   beta_1=PARAMS['beta_1'],
                   beta_2=PARAMS['beta_2'],
                   epsilon=PARAMS['epsilon'],
                   decay=PARAMS['decay']
  )
elif PARAMS['optimizer'] == 'SGD':
  optimizer = SGD(lr=PARAMS['learning_rate'],
                  decay=PARAMS['decay'],
                  momentum=PARAMS['momentum'],
                  nesterov=True
  )
#
BATCH_SIZE = PARAMS['batch_size']
epochs = PARAMS['epochs']
custLossThresh = PARAMS['custLossThresh']
label_level = PARAMS['label_level']
label_feature_thresh = PARAMS['label_feature_threshold']
model_run_name = PARAMS['model_run_name']
feature_description = PARAMS['feature_description']
#
load_previous_model = False
previous_model = 'trained_model_feature-0to6.5km_at_500m_steps_label-5500m_2020_11_19_17_32.h5'
input_model = os.path.join(output_root_directory,model_run_name, previous_model)
#
#--------------------------
output_model_name     = 'trained_model_feature-' + feature_description + '_{}.h5'
log_dir = os.path.join(output_root_directory, model_run_name, 'logs', 'fit',output_model_name.format(t))
#
data_fraction_for_training = 0.65
data_fraction_for_validation = 0.25
#
############################

output_path = os.path.join(output_root_directory, model_run_name)
if not os.path.exists(output_path):
    os.makedirs(output_path)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

#
# load the data
#
fds = xr.open_dataset(feature_data)
lds = xr.open_dataset(label_data)
feature = fds.REFL_10CM.values

#
# read composite of W
#
label = lds.W.values
label = label[:,np.newaxis,:,:]

#
# move the channels from position 1 to position 3
# goes from [time,channel,height,width] to [time, height, width, channel]
# which is the default for Conv2D.
#
feature = np.moveaxis(feature, 1, 3)
label = np.moveaxis(label, 1, 3)
# 
feature2D = np.max(feature, axis=3)
#feature2D = feature[:,:,:,0]
feature2D = feature2D[:,:,:,np.newaxis]
#
#
# Based on the min and max from scaling the data
# V = Vs * (45.6660 - -35.0) + -35.0
#   = Vs * 80.6660 + -35.0
# a value of 0.001 would be ~ -34.92dbz
# a value of 0.5 would be ~05dbz
# a value of 0.6 would be ~13dbz
# a value of 0.7 would be ~21dbz
# a value of 0.8 would be ~30dbz
# 
#for i in range(label.shape[3]):
#    label[:,:,:,i] = label[:,:,:,i][feature2D[:,:,:,0]<0.4] = -99.0
#
label[feature2D<label_feature_thresh] = custLossThresh
#
# using percentages to get the start and end indexes for 
# the training and the validation datasets
#
num_images = feature.shape[0]

train_data_start = 0
train_data_end   = int( num_images * data_fraction_for_training  / BATCH_SIZE ) * BATCH_SIZE

val_data_start = train_data_end + 1
val_data_end = int(  ( num_images * (data_fraction_for_training + data_fraction_for_validation) - val_data_start)  / BATCH_SIZE )
val_data_end = (val_data_end * BATCH_SIZE) + val_data_start

print ()
print ("Number of images:", num_images)
print ("Training data start image:", train_data_start)
print ("Training data end image:", train_data_end)
print ("Validation data start image:", val_data_start)
print ("Validation data end image:", val_data_end)
print ()

SHUFFLE_BUFFER_SIZE = train_data_end

#
# If reading in the 3D label data
#
#train_dataset = tf.data.Dataset.from_tensor_slices((feature[train_data_start:train_data_end,:,:,:], label[train_data_start:train_data_end,:,:,label_level]))
#val_dataset   = tf.data.Dataset.from_tensor_slices((feature[val_data_start:val_data_end,:,:,:], label[val_data_start:val_data_end,:,:,label_level]))
#
# If reading in just one level for the label data
#
train_dataset = tf.data.Dataset.from_tensor_slices((feature[train_data_start:train_data_end,:,:,:], label[train_data_start:train_data_end,:,:,:]))
val_dataset   = tf.data.Dataset.from_tensor_slices((feature[val_data_start:val_data_end,:,:,:], label[val_data_start:val_data_end,:,:,:]))
#
# 
#
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True).batch(BATCH_SIZE, drop_remainder=True)
val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

#
# set up the model
#
output_model = os.path.join(output_path, output_model_name)
#
if load_previous_model:
  model = tf.keras.models.load_model(input_model, compile=False)
else:
  model = unet()
#
model.compile(optimizer=optimizer, loss=cust_loss(custLossThresh), metrics = ['mae',cust_accuracy(custLossThresh),cust_mae(custLossThresh),dice_coef], run_eagerly=True)

#
# set up callback functions
#
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(output_path,"trained_weights_best.h5"), monitor=cust_loss(custLossThresh), verbose=1, save_best_only=True, mode='min')
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(output_path,"trained_weights_best.h5"), monitor='loss', verbose=1, save_best_only=True, mode='min', save_freq='epoch')

#
# Adam optimizer adaptively computes updates to the learning rate
# so scheduler is taken out for this optimizer
#
LRS = tf.keras.callbacks.LearningRateScheduler(scheduler)

#
# run the model 
#
neptune_tb.integrate_with_tensorflow()
model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[tensorboard, checkpoint, NeptuneMonitor()])

#
# Send signal to neptune that the run is done
#
neptune.stop()

#
# save the model from the last epoch
#
t = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
model.save(output_model.format(t))
