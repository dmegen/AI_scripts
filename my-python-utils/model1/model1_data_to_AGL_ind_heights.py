#!/usr/bin/env python

import xarray as xr
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import load_data
import wrf

def downscale_remove(feature, label):
    
    j=0
    a = np.zeros((len(feature)*4,128,128))
    b = np.zeros((len(feature)*4,128,128))
    
    for i in range(len(feature) - 1):
        if np.amax(feature[i,0:128,0:128]) != np.amin(feature[i,0:128,0:128]):
            a[j,:,:] = feature[i,0:128,0:128]
            b[j,:,:] = label[i,0:128,0:128]
            j = j+1
        if np.amax(feature[i,0:128,128:256]) != np.amin(feature[i,0:128,128:256]):
            a[j,:,:] = feature[i,0:128,128:256]
            b[j,:,:] = label[i,0:128,128:256]
            j = j+1
        if np.amax(feature[i,128:256,0:128]) != np.amin(feature[i,128:256,0:128]):
            a[j,:,:] = feature[i,128:256,0:128]
            b[j,:,:] = label[i,128:256,0:128]
            j = j+1
        if np.amax(feature[i,128:256,128:256]) != np.amin(feature[i,128:256,128:256]):
            a[j,:,:] = feature[i,128:256,128:256]
            b[j,:,:] = label[i,128:256,128:256]
            j = j+1

    print(j)

    return np.resize(a, (j,128,128)), np.resize(b, (j,128,128))

def interp_data(fieldname, field, Z_AGL):
    
    interp_levels = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000]
    #interp_levels = [2000]
    
    vars = {}    
    for level in interp_levels:

        varname = fieldname + "_" + str(level) + "m_AGL"
    
        if level == 0:
            vars[varname] = field[:,0,:,:]
        else:
            #
            # interpolate to the requested level
            # 
            vars[varname] = wrf.interplevel(field[:,:,:,:], Z_AGL[:,:,:,:], level, meta=False)
        
    return vars

#
# Load the model label field (W)
#
#w = load_data.load_W_data_oneTime()
w = load_data.load_W_data_all(slevel=0, elevel=50, method="sel")
print(w.shape)

#
# Load the model feature (QRAIN)
#
#qr = load_data.load_QRAIN_data_oneTime()
qr = load_data.load_QRAIN_data_all(slevel=0, elevel=50, method="sel")
print(qr.shape)

#
# Load another model feature field (QSNOW)
#
#qs = load_data.load_QSNOW_data_all(slevel=0, elevel=50, method="sel")
#print(qs.shape)

#
# Load the Z data and get the terrain height using the 
# models lowest Z level (hybrid level data)
#
#z = load_data.load_Z_data_oneTime()
z = load_data.load_Z_data_all(slevel=0, elevel=50, method='sel')
terrain_height = z[:,0,:,:].copy()
terrain_height = terrain_height[:,np.newaxis,:,:]
print(z.shape)
print(terrain_height.shape)

#
# Convert the Z data from MSL to AGL
#
#for i in range(0,z.shape[1]):
#    z[:,i,:,:] = z[:,i,:,:] - terrain_height[:,0,:,:]
z = z - terrain_height
print(z.shape)

#
# interpolated the W field to the requested
# heights AGL
#
W_VARS = interp_data("W", w, z)
for key in W_VARS.keys() :
    print (key, W_VARS[key].shape)
del w

#
# interpolated the QRAIN field to the requested
# heights AGL 
#
#QR_VARS = interp_data("QR", qr, z)
#for key in QR_VARS.keys() :
#    print (key, QR_VARS[key].shape)
#del qr

#
# interpolated the QSNOW field to the requested
# heights AGL and use min/max scaling to
# scale the data to values between 0 and 1
#
QR_VARS = interp_data("QR", qr, z)
for key in QR_VARS.keys() :
    print (key, QR_VARS[key].shape)
del qs

#
# downscale the data and remove images where
# the feature data is all one value
# write each AGL level to a file
#
for W_key, QR_key in zip(W_VARS.keys(), QR_VARS.keys()):
    
    print(W_key, W_VARS[W_key].shape)
    print(QR_key, QR_VARS[QR_key].shape)
    
    DS_feature, DS_label = downscale_remove(QR_VARS[QR_key], W_VARS[W_key])
    
    #DS_feature = (DS_feature - np.amin(DS_feature)) / (np.amax(DS_feature) - np.amin(DS_feature))
    #DS_label = (DS_label - np.amin(DS_label)) / (np.amax(DS_label) - np.amin(DS_label))
    
    for i in range(len(DS_feature) - 1):
        if np.amax(DS_feature[i,:,:]) == np.amin(DS_feature[i,:,:]):
            DS_feature[i,:,:] = 0
        else:
            DS_feature[i,:,:] = (DS_feature[i,:,:] - np.amin(DS_feature[i,:,:])) / (np.amax(DS_feature[i,:,:]) - np.amin(DS_feature[i,:,:]))
        if np.amax(DS_label[i,:,:]) == np.amin(DS_label[i,:,:]):
            DS_label[i,:,:] = 0
        else:
            DS_label[i,:,:] = (DS_label[i,:,:] - np.amin(DS_label[i,:,:])) / (np.amax(DS_label[i,:,:]) - np.amin(DS_label[i,:,:]))
        
        
    #
    # write downscaled feature_data
    #
    output_data = '/glade/work/hardt/ds612/2000-2013_June-Sept_DS128_'+ QR_key + '.nc'
    QR = xr.DataArray(DS_feature, name='QRAIN')
    encoding={'QRAIN': {'zlib': True, '_FillValue': -99.0}}
    QR.to_netcdf(output_data, encoding=encoding)
    
    #
    # write downscaled label_data
    #
    output_data = '/glade/work/hardt/ds612/2000-2013_June-Sept_DS128_QSNOW_' + W_key + '.nc'
    WAGL = xr.DataArray(DS_label, name='W')
    encoding={'W': {'zlib': True, '_FillValue': -99.0}}
    WAGL.to_netcdf(output_data, encoding=encoding)
    