#!/usr/bin/env python

import neptune
import neptune_tensorboard as neptune_tb
from neptunecontrib.monitoring.keras import NeptuneMonitor

import optuna

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

def thresh_loss(y_true, y_pred, thresh):
  y_true = y_true[y_true > thresh]
  y_pred = y_pred[y_true > thresh]
  mse = tf.keras.losses.MeanSquaredError()
  huber = tf.keras.losses.Huber()
  return huber(y_true, y_pred)

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

def objective(trial):
    optuna_params = {'batch_size': trial.suggest_categorical('batch_size', [32, 64]),
                     'activation': trial.suggest_categorical('activation', ['relu', 'elu']),
                     'learning_rate': trial.suggest_loguniform('learning_rate', 0.0001, 0.1),
                     'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'Nadam', 'SGD']),
                     'dense_units': trial.suggest_categorical('dense_units', [16, 32, 64, 128]),
                     'dropout': trial.suggest_uniform('dropout', 0, 0.5),
                     }
    PARAMS = {**optuna_params, **STATIC_PARAMS}
    return train_evaluate(PARAMS)

############################
# set up the run information
############################

neptune.init(project_qualified_name='hardt/Predicting-W',
             api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiOTFlYTI0ZDAtMzk3Zi00NzU0LWEzNWUtMzY2ZDcxZTA4OTg2In0=')
neptune_tb.integrate_with_tensorflow()

STATC_PARAMS = {'epochs': 10,
                'beta_1': 0.9,
                'beta_2': 0.999,
                'epsilon': 1.0,
                'decay': 0.0,
                'momentum': 0.9,
          }
PARAMS = {'epochs': 10,
          'batch_size': 32,
          'optimizer': 'Adam',
          'learning_rate': 0.01,
          'beta_1': 0.9,
          'beta_2': 0.999,
          'epsilon': 1.0,
          'decay': 0.0,
          'momentum': 0.9,
          }

# make optuna study
#study = optuna.create_study(study_name='classification',
#                            direction='maximize',
#                            storage='sqlite:///classification.db',
#                            load_if_exists=True)
#study.optimize(objective, n_trials=100)

# run experiment that collects study visuals

neptune.create_experiment(name='v5p0 14channel-W-5.5km 0-100 epochs', 
                          params=PARAMS,
                          tags=['Adam', 'LRS', 'Shuffle'],
                          id='v5p0-1'
)


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

output_root_directory = '/glade/work/hardt/models'
model_run_name        = 'unet_v5p0'
from unet_model_v4p0 import unet

#
# Altitude in meters to run
#
feature_description = '0to6.5km_at_500m_steps'

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
level_count = 1
for i in range(0,7500,500):
    label_name = str(i)
    levels[label_name] = level_count
    level_count = level_count + 1

level_label = '5500'
label_level = levels[level_label]
#--------------------------

load_previous_model = False
previous_model = 'trained_model_feature-0to6.5km_at_500m_steps_label-5500m_2020_11_19_17_32.h5'
input_model = os.path.join(output_root_directory,model_run_name, previous_model)

#--------------------------

output_model_name     = 'trained_model_feature-' + feature_description + '_label-' + level_label + 'm_{}.h5'
log_dir = os.path.join(output_root_directory, model_run_name, 'logs', 'fit',output_model_name.format(t))
feature_data          = '/glade/work/hardt/ds612/2000-2013_June-Sept_QRAIN_INTERP_AGL_0to7km_at_500m_steps.nc'
label_data            = '/glade/work/hardt/ds612/2000-2013_June-Sept_W_INTERP_AGL_0to7km_at_500m_steps.nc'

BATCH_SIZE = PARAMS['batch_size']
epochs = PARAMS['epochs']

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
# random shuffle 
#
s = np.arange(feature.shape[0])
np.random.shuffle(s)

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
# What to do if I only want only one level 
# from the feature data. Can not be done as
# a tensor.
#
# feature = feature[:,:,:,label_level]
# print(feature.shape)
# And set feature[train_data_start:train_data_end,:,:,np.newaxis]
# feature data needs the channels dimension but the label does not.
#

#
# set up the the data sets
#
# change unet to (256,256,1)
#
# train_dataset = tf.data.Dataset.from_tensor_slices((feature[train_data_start:train_data_end,:,:,np.newaxis], label[train_data_start:train_data_end,:,:,label_level]))
# val_dataset   = tf.data.Dataset.from_tensor_slices((feature[val_data_start:val_data_end,:,:,np.newaxis], label[val_data_start:val_data_end,:,:,label_level]))
#
# change unet to (256,256,14)
# was getting Nan's for loss function when I included level 15 (7km's)
#
train_dataset = tf.data.Dataset.from_tensor_slices((feature[s][train_data_start:train_data_end,:,:,:14], label[s][train_data_start:train_data_end,:,:,label_level]))
val_dataset   = tf.data.Dataset.from_tensor_slices((feature[s][val_data_start:val_data_end,:,:,:14], label[s][val_data_start:val_data_end,:,:,label_level]))
#
#
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
model.compile(optimizer=optimizer, loss=cust_loss(0.01), metrics = ['accuracy','mae'], run_eagerly=True)
#model.compile(optimizer = Adam(lr=1e-6, beta_1=0.9, beta_2=0.999, epsilon=1.0, decay=0.0), loss=mse, metrics = ['accuracy','mae'], run_eagerly=True)

#
# callbacks
#
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model_save_callback = tf.keras.callbacks.ModelCheckpoint(filepath='/glade/scratch/hardt/unet_v1/trained_model_epoch{epoch}.h5',save_freq='epoch')
#checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(output_path,"trained_weights_best_epoch{epoch}.h5"), monitor='mae', verbose=1, save_best_only=True, mode='min')
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(output_path,"trained_weights_best_" + level_label + "AGL.h5"), monitor='mae', verbose=1, save_best_only=True, mode='min')
LRS = tf.keras.callbacks.LearningRateScheduler(scheduler)

#
# do the training
#
#model.fit(train_dataset, epochs=100, validation_data=val_dataset, callbacks=[tensorboard_callback, LRS, checkpoint])
model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[tensorboard, LRS, checkpoint, NeptuneMonitor()])
# model.fit(train_dataset, epochs=100,
          # Only run validation using the first 10 batches of the dataset
          # using the `validation_steps` argument
#          validation_data=val_dataset, validation_steps=10)

#
# write out the trained model
#
t = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
model.save(output_model.format(t))

#neptune.log_metric('optuna_best_score', study.best_value)
#neptune.set_property('optuna_best_parameters', study.best_params)
#neptune.log_artifact('classification.db', 'classification.db')
neptune.stop()
