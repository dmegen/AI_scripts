#!/usr/bin/env python

import xarray as xr
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import load_data
import wrf

def downscale(feature):
    
    a = np.zeros((feature.shape[0]*4,feature.shape[1],128,128))
    
    j = 0
    for i in range(feature.shape[0]):
        for v in range(feature.shape[1]):
            a[j,v,:,:] = feature[i,v,0:128,0:128]
            
            a[j+1,v,:,:] = feature[i,v,0:128,128:256]
 
            a[j+2,v,:,:] = feature[i,v,128:256,0:128]
        
            a[j+3,v,:,:] = feature[i,v,128:256,128:256]
            
        j = j + 4

    print("number of 128x128 images:", j)
    return a

def downscale_2(feature, label):
    
    a = np.zeros((feature.shape[0]*4,feature.shape[1],128,128))
    b = np.zeros((feature.shape[0]*4,feature.shape[1],128,128))
    
    j = 0
    for i in range(feature.shape[0]):
        for v in range(feature.shape[1]):
            a[j,v,:,:] = feature[i,v,0:128,0:128]
            b[j,v,:,:] = label[i,v,0:128,0:128]
            
            a[j+1,v,:,:] = feature[i,v,0:128,128:256]
            b[j+1,v,:,:] = label[i,v,0:128,128:256]
 
            a[j+2,v,:,:] = feature[i,v,128:256,0:128]
            b[j+2,v,:,:] = label[i,v,128:256,0:128]
        
            a[j+3,v,:,:] = feature[i,v,128:256,128:256]
            b[j+3,v,:,:] = label[i,v,128:256,128:256]
            
        j = j + 4

    print("number of 128x128 images:", j)
    return a,b

def downscale_remove(feature, label):
    
    a = np.zeros((feature.shape[0]*4,feature.shape[1],128,128))
    b = np.zeros((feature.shape[0]*4,feature.shape[1],128,128))
    
    j = 0
    for i in range(feature.shape[0]):
        for v in range(feature.shape[1]):
            if (np.amax(feature[i,v,0:128,0:128]) != np.amin(feature[i,v,0:128,0:128]) and
                np.amax(feature[i,v,0:128,128:256]) != np.amin(feature[i,v,0:128,128:256]) and
                np.amax(feature[i,v,128:256,0:128]) != np.amin(feature[i,v,128:256,0:128]) and
                np.amax(feature[i,v,128:256,128:256]) != np.amin(feature[i,v,128:256,128:256]) ):
                    a[j,v,:,:] = feature[i,v,0:128,0:128]
                    b[j,v,:,:] = label[i,v,0:128,0:128]
            
                    a[j+1,v,:,:] = feature[i,v,0:128,128:256]
                    b[j+1,v,:,:] = label[i,v,0:128,128:256]
 
                    a[j+2,v,:,:] = feature[i,v,128:256,0:128]
                    b[j+2,v,:,:] = label[i,v,128:256,0:128]
        
                    a[j+3,v,:,:] = feature[i,v,128:256,128:256]
                    b[j+3,v,:,:] = label[i,v,128:256,128:256]
                    
        j = j + 4
                   
                
    print("number of 128x128 images:", j)
    for i in range(j,feature.shape[0]):
        a = np.delete(a, j, 0)
        b = np.delete(b, j, 0)

    #return np.resize(a, (j,min_nv,128,128), np.resize(b, (j,min_nv,128,128))
    return a,b

def interp_data_concat(field, Z_AGL):
    
    interp_levels = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000]
    
    count = 0
    for level in interp_levels:
        
        count = count + 1
        
        if level == 0:
            interp_field = field[:,0,:,:]
            interp_field = interp_field[:,np.newaxis,:,:]
        else:
            #
            # interpolate to the requested level
            # 
            interp_field = wrf.interplevel(field[:,:,:,:], Z_AGL[:,:,:,:], level, meta=False)
            interp_field = interp_field[:, np.newaxis,:,:]
        
        if count == 1:
            interp_3d = interp_field
            print(interp_3d.shape)
        else:
            interp_3d = np.concatenate( (interp_3d, interp_field), axis=1)
            print(interp_3d.shape)
            
    return interp_3d

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
del terrain_height

#
# Load the model label field (W)
#
#w = load_data.load_W_data_oneTime()
w = load_data.load_W_data_all(slevel=0, elevel=50, method="sel")
print(w.shape)
#
# interpolated the W field to the requested
# heights AGL
#
W_VAR = interp_data_concat(w, z)
del w
#
# Load the model feature (QRAIN)
#
#qr = load_data.load_QRAIN_data_oneTime()
qr = load_data.load_QRAIN_data_all(slevel=0, elevel=50, method="sel")
print(qr.shape)
#
# interpolated the QRAIN field to the requested
# heights AGL 
#
QR_VAR = interp_data_concat(qr, z)
del qr
#
DS_feature, DS_label = downscale_2(QR_VAR, W_VAR)
#
del QR_VAR
del W_VAR
#
for i in range(DS_feature.shape[0] - 1):
    for v in range(DS_feature.shape[1] - 1):
        if np.amax(DS_feature[i,v,:,:]) == np.amin(DS_feature[i,v,:,:]):
            DS_feature[i,v,:,:] == 0
        else:
            DS_feature[i,v,:,:] = (DS_feature[i,v,:,:] - np.amin(DS_feature[i,v,:,:])) / (np.amax(DS_feature[i,v,:,:]) - np.amin(DS_feature[i,v,:,:]))
#
output_data = '/glade/work/hardt/ds612/2000-2013_June-Sept_DS128_QRAIN_INTERP_AGL_0to7km_at_500m_steps.nc'
QR = xr.DataArray(DS_feature, name='QRAIN')
encoding={'QRAIN': {'zlib': True, '_FillValue': -99.0}}
QR.to_netcdf(output_data, encoding=encoding)
del DS_feature
#
for i in range(DS_label.shape[0] - 1):
    for v in range(DS_label.shape[1] - 1):
        if np.amax(DS_label[i,v,:,:]) == np.amin(DS_label[i,v,:,:]):
            DS_label[i,v,:,:] == 0
        else:
            DS_label[i,v,:,:] = (DS_label[i,v,:,:] - np.amin(DS_label[i,v,:,:])) / (np.amax(DS_label[i,v,:,:]) - np.amin(DS_label[i,v,:,:]))

output_data = '/glade/work/hardt/ds612/2000-2013_June-Sept_DS128_QR_W_INTERP_AGL_0to7km_at_500m_steps.nc'
WAGL = xr.DataArray(DS_label, name='W')
encoding={'W': {'zlib': True, '_FillValue': -99.0}}
WAGL.to_netcdf(output_data, encoding=encoding)
del DS_label
#
# Load another model feature field (QSNOW)
#
qs = load_data.load_QSNOW_data_all(slevel=0, elevel=50, method="sel")
print(qs.shape)
#
# interpolated the QSNOW field to the requested
# heights AGL
#
QS_VAR = interp_data_concat(qs, z)
del qs
#
DS128_feature = downscale(QS_VAR)
#
for i in range(DS128_feature.shape[0] - 1):
    for v in range(DS128_feature.shape[1] - 1):
        if np.amax(DS128_feature[i,v,:,:]) == np.amin(DS128_feature[i,v,:,:]):
            DS128_feature[i,v,:,:] == 0
        else:
            DS128_feature[i,v,:,:] = (DS128_feature[i,v,:,:] - np.amin(DS128_feature[i,v,:,:])) / (np.amax(DS128_feature[i,v,:,:]) - np.amin(DS128_feature[i,v,:,:]))
#
output_data = '/glade/work/hardt/ds612/2000-2013_June-Sept_DS128_QSNOW_INTERP_AGL_0to7km_at_500m_steps.nc'
QS = xr.DataArray(DS128_feature, name='QSNOW')
encoding={'QSNOW': {'zlib': True, '_FillValue': -99.0}}
QS.to_netcdf(output_data, encoding=encoding)
#
del DS128_feature
#
#for i in range(DS128_label.shape[0] - 1):
#    for v in range(DS128_label.shape[1] - 1):
#        if np.amax(DS128_label[i,v,:,:]) == np.amin(DS128_label[i,v,:,:]):
#            DS128_label[i,v,:,:] == 0
#        else:
#            DS128_label[i,v,:,:] = (DS128_label[i,v,:,:] - np.amin(DS128_label[i,v,:,:])) / (np.amax(DS128_label[i,v,:,:]) - np.amin(DS128_label[i,v,:,:]))

#output_data = '/glade/work/hardt/ds612/2000-2013_June-Sept_DS128_QS_W_INTERP_AGL_0to7km_at_500m_steps.nc'
#WAG = xr.DataArray(DS128_label, name='W')
#encoding={'W': {'zlib': True, '_FillValue': -99.0}}
#WAG.to_netcdf(output_data, encoding=encoding)
#del DS128_label
