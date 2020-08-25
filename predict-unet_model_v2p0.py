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

model_run_name   = 'unet_v2p0'
model            = 'trained_weights_best_job_5685240.h5'
                    
output_data_name = 'predict_' + model + '.nc'

feature_data     = '/glade/work/hardt/ds612/2000-2013_June-Sept_DS128_REFL.nc'
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
feature = fds.refl.values
test_dataset = tf.data.Dataset.from_tensor_slices((feature[22306:27874,:,:,np.newaxis]))

BATCH_SIZE = 32
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)

#
# run the model predictions
#
model = tf.keras.models.load_model(input_model, compile=False)
y0n = model.predict(test_dataset)

#
# write perdiction data to netcdf file
#
pds = xr.DataArray(y0n, name='pMaxW')
encoding={'pMaxW': {'zlib': True, '_FillValue': -9999.0}}
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

