#!/usr/bin/env python

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
  if epoch < 6:
    return 0.0001
  else:
    return 0.0001 * tf.math.exp(0.1 * (10 - epoch))

def thresh_loss(y_true, y_pred, thresh):
  mask = tf.math.greater(y_true, thresh)
  y_true2 = tf.boolean_mask(y_true, mask)
  y_pred2 = tf.boolean_mask(y_pred, mask)
  mse = tf.keras.losses.MeanSquaredError()
  huber = tf.keras.losses.Huber()
  return huber(y_true2, y_pred2)

def cust_loss(thresh):
  def loss(y_true, y_pred):
    return thresh_loss(y_true, y_pred, thresh)
  return loss

#
# Mean Absolute Error metric
#
def mae(y_true, y_pred):
  eval = K.abs(y_pred - y_true)
  eval = K.mean(eval, axis=-1)
  return eval

############################
# set up the run information
############################

output_root_directory = '/glade/work/hardt/models'
model_run_name        = 'unet_v4p1'
from unet_model_v4p1 import unet

#
# Altitude in meters to run
#
ALT = '0to7km_at_500m_steps'

# 
# 1)     0 meters AGL
# 2)   500
# 3)  1000 
# 4)  1500 
# 5)  2000
# 6)  2500
# 7)  3000
# 15) 7000 meters AGL
#
levels = {}
level_count = 0
for i in range(0,7000,500):
    label_name = str(i)
    levels[label_name] = level_count
    level_count = level_count + 1

label_level = levels['2000']
#--------------------------

load_previous_model = False
previous_model = 'trained_model_2000m_2020_09_10_11_36.h5'
input_model = os.path.join(output_root_directory,model_run_name, previous_model)

#--------------------------

output_model_name     = 'trained_model_' + str(ALT) +'m_{}.h5'
log_dir = os.path.join(output_root_directory, model_run_name, 'logs', 'fit',output_model_name.format(t))
feature_data          = '/glade/work/hardt/ds612/2000-2013_June-Sept_DS128_QRAIN_INTERP_AGL_' + str(ALT) + '.nc'
label_data            = '/glade/work/hardt/ds612/2000-2013_June-Sept_DS128_W_INTERP_AGL_' + str(ALT) + '.nc'

BATCH_SIZE = 32
epochs = 5

data_fraction_for_training = 0.65
data_fraction_for_validation = 0.25

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
feature = fds.QRAIN.values
label = lds.W.values

#
# move the channels from position 1 to position 3
# goes from [time,channel,height,width] to [time, height, width, channel]
# which is the default for Conv2D.
#
feature = np.moveaxis(feature, 1, 3)
label = np.moveaxis(label, 1, 3)

#
# 
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
print ("Valication data start image:", val_data_start)
print ("Validation data end image:", val_data_end)
print ()

SHUFFLE_BUFFER_SIZE = train_data_end

#
# set up the the data sets
#
train_dataset = tf.data.Dataset.from_tensor_slices((feature[train_data_start:train_data_end,:,:,:], label[train_data_start:train_data_end,:,:,label_level]))
val_dataset   = tf.data.Dataset.from_tensor_slices((feature[val_data_start:val_data_end,:,:,:], label[val_data_start:val_data_end,:,:,label_level]))

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True).batch(BATCH_SIZE, drop_remainder=True)
val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

#
# set up the model
#
output_model = os.path.join(output_path, output_model_name)

if load_previous_model:
  model = tf.keras.models.load_model(input_model, compile=False)
else:
  model = unet()

mse = tf.keras.losses.MeanSquaredError()
#model.compile(optimizer = SGD(lr=1e-6, momentum=0.5), loss=mse, metrics = ['accuracy'], run_eagerly=True)
model.compile(optimizer = Adam(lr=1e-6, beta_1=0.9, beta_2=0.999, epsilon=1.0, decay=0.0), loss=cust_loss(0.01), metrics = ['accuracy','mae'], run_eagerly=True)
#model.compile(optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1.0, decay=0.0), loss = 'mse', metrics = ['accuracy'], run_eagerly=True)

#
# callbacks
#
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model_save_callback = tf.keras.callbacks.ModelCheckpoint(filepath='/glade/scratch/hardt/unet_v1/trained_model_epoch{epoch}.h5',save_freq='epoch')
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(output_path,"trained_weights_best_epoch{epoch}.h5"), monitor='mae', verbose=1, save_best_only=True, mode='min')
LRS = tf.keras.callbacks.LearningRateScheduler(scheduler)

#
# do the training
#
#model.fit(train_dataset, epochs=100, validation_data=val_dataset, callbacks=[tensorboard_callback, LRS, checkpoint])
model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[tensorboard_callback, checkpoint])
# model.fit(train_dataset, epochs=100,
          # Only run validation using the first 10 batches of the dataset
          # using the `validation_steps` argument
#          validation_data=val_dataset, validation_steps=10)

#
# write out the trained model
#
t = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
model.save(output_model.format(t))
