#!/usr/bin/env python

import tensorflow as tf
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

############################
# set up the run information
############################
root_directory   = '/glade/work/hardt/models'
#
model_run_name   = 'unet_v6p0'
#
model            = 'trained_model_feature-00minAfterHour_refl_2021_01_26_10_21.h5'
#
output_data_name = 'predict-test-' + model + '.nc'
#output_data_name = 'predict-train-' + model + '.nc'
#
feature_data     = '/glade/work/hardt/ds612/model2_00minuteAfterHour_3D_refl_shuffled_scaled-v6.nc'
#
BATCH_SIZE = 32
#
data_fraction_for_test = 0.1
#
############################

model_run_dir = os.path.join(root_directory, model_run_name)
output_data = os.path.join(model_run_dir, output_data_name)
input_model = os.path.join(model_run_dir, model)

if not os.path.isfile(input_model):
    print("ERROR: Missing input model file", input_model)
    sys.exit()

#
# set up the test data set
#
fds = xr.open_dataset(feature_data)
feature = fds.REFL_10CM.values
#
num_images = feature.shape[0]
#
test_data_start = int(num_images * (1 - data_fraction_for_test))
test_data_start = (num_images - int((num_images - test_data_start) / BATCH_SIZE) * BATCH_SIZE) 
test_data_end = num_images
#
print ()
print ("Number of images:", num_images)
print ("Test data start image:", test_data_start)
print ("Test data end image:", test_data_end)
#
# move the channels from position 1 to position 3
# goes from [time,channel,height,width] to [time, height, width, channel]
# which is the default for Conv2D.
#
feature = np.moveaxis(feature, 1, 3)

test_dataset = tf.data.Dataset.from_tensor_slices((feature[test_data_start:test_data_end,:,:,:]))
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)

#
# run the model predictions
#
model = tf.keras.models.load_model(input_model, compile=False)
y0n = model.predict(test_dataset)

#
# write perdiction data to netcdf file
#
pds = xr.DataArray(y0n, name='pW')
encoding={'pW': {'zlib': True, '_FillValue': -9999.0}}
pds.to_netcdf(output_data, encoding=encoding)

#
# save some images
# 

#for i in range(1000,1020):
#    plt.clf()
#    plt.imshow(y0n[i,:,:,0])
#    plt.colorbar()
#    plt.savefig(model_run_dir + "/pMaxW_s1_{:04}.png".format(i))

#max_data = y0n.max(axis=1).max(axis=1)[:,0]

#plt.clf()
#plt.plot(max_data)
#plt.savefig(model_run_dir + "/max_time_series_s1.png")

#best = np.argmax(max_data)

#plt.clf()
#plt.imshow(y0n[best,:,:,0])
#plt.colorbar()
#plt.savefig(model_run_dir + "/best_s1.png")

