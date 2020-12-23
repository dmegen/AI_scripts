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
    y_true[y_true!=thresh]
    y_pred[y_true!=thresh]
    mse = tf.keras.losses.MeanSquaredError()
    huber = tf.keras.losses.Huber()
    return mse(y_true, y_pred)

def cust_loss(thresh):
    def loss(y_true, y_pred):
        return custom_loss(y_true, y_pred, thresh)
    return loss
    
#
# Mean Absolute Error metric
#
def mae(y_true, y_pred):
  eval = K.abs(y_pred - y_true)
  eval = K.mean(eval, axis=-1)
  return eval

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    coef = (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
    return coef

############################
# set up the run information
############################

PARAMS = {'epochs': 100,
          'batch_size': 32,
          'optimizer': 'Adam',
          'learning_rate': 0.01,
          'beta_1': 0.9,
          'beta_2': 0.999,
          'epsilon': 1.0,
          'decay': 0.0,
          'momentum': 0.9,
          'custLossThresh': -99.0,
          'refl_scaling_min': -35.0,
          'refl_scaling_per99.99': 58.3864573,
          'W_scaling_min': -13.606483,
          'W_scaling_per99.99': 1.2770988,
          }

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

#--------------------------

output_root_directory = '/glade/work/hardt/models'
model_run_name        = 'unet_v6p0'
from unet_model_v6p0 import unet

#
# Altitude in meters to run
#
feature_description = '5minAfterHour_refl'

#--------------------------

load_previous_model = False
previous_model = 'trained_model_feature-0to6.5km_at_500m_steps_label-5500m_2020_11_19_17_32.h5'
input_model = os.path.join(output_root_directory,model_run_name, previous_model)

#--------------------------

output_model_name     = 'trained_model_feature-' + feature_description + '_{}.h5'
log_dir = os.path.join(output_root_directory, model_run_name, 'logs', 'fit',output_model_name.format(t))
feature_data          = '/glade/work/hardt/ds612/model2_0minuteAfterHour_3D_refl_shuffled.nc'
label_data            = '/glade/work/hardt/ds612/model2_3D_W_shuffled.nc'

BATCH_SIZE = PARAMS['batch_size']
epochs = PARAMS['epochs']
custLossThresh = PARAMS['custLossThresh']

data_fraction_for_training = 0.65
data_fraction_for_validation = 0.25

#
# This data has 51 levels, so a value between 0-50
#
label_level = 1
#
# Based on the min and max from scaling the data
# V = Vs * (refl_scaling_per99.99 - refl_scaling_min) - refl_scaling_min
#   = Vs * 93.3865 + -35
# a value of 0.4 would be ~02dbz
# a value of 0.5 would be ~11dbz
# a value of 0.6 would be ~20dbz
# a value of 0.7 would be ~30dbz
# 
label_feature_thresh = 0.001

#
# Set up neptune
#
NEPTUNE_API_TOKEN = os.environ.get('NEPTUNE_API_TOKEN')
neptune.init(project_qualified_name='hardt/Pred-W-RefOffset',
             api_token=NEPTUNE_API_TOKEN)
neptune.create_experiment(name='v6p0 ref0minAfter Wmax thresh 0.001', 
                          params=PARAMS,
                          tags=['v6p0', 'Adam', 'LRS', 'Shuffle','CustomLoss', 'unet']
)

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
# get all levels 
#
#label = lds.W.values
#
# Get just one level
#
#label = lds.W[:,label_level,:,:].values
#label = label[:,np.newaxis,:,:]
#
# Get the max value in the vertical
#
label = lds.W.values.max(axis=1)
label = label[:,np.newaxis,:,:]

#
# move the channels from position 1 to position 3
# goes from [time,channel,height,width] to [time, height, width, channel]
# which is the default for Conv2D.
#
feature = np.moveaxis(feature, 1, 3)
label = np.moveaxis(label, 1, 3)

#
# Based on the min and max from scaling the data
# V = Vs * (refl_scaling_per99.99 - refl_scaling_min) - refl_scaling_min
#   = Vs * 93.3865 + -35
# a value of 0.5 would be ~11dbz
# a value of 0.6 would be ~20dbz
# a value of 0.7 would be ~30dbz
# 
# 
#feature2D = np.amax(feature, axis=3)
feature2D = feature[:,:,:,0]
feature2D = feature2D[:,:,:,np.newaxis]
#
#for i in range(label.shape[3]):
#    label[:,:,:,i] = label[:,:,:,i][feature2D[:,:,:,0]<0.4] = -99.0
    
label[feature2D<label_feature_thresh] = -99.0

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
model.compile(optimizer=optimizer, loss=cust_loss(custLossThresh), metrics = ['accuracy','mae',dice_coef], run_eagerly=True)

#
# set up callback functions
#
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(output_path,"trained_weights_best.h5"), monitor='mae', verbose=1, save_best_only=True, mode='min')
LRS = tf.keras.callbacks.LearningRateScheduler(scheduler)

#
# run the model 
#
model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[tensorboard, LRS, checkpoint, NeptuneMonitor()])

#
# Send signal to neptune that the run is done
#
neptune.stop()

#
# save the model from the last epoch
#
t = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
model.save(output_model.format(t))