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
    min_nv = 10000
    a = np.zeros((feature.shape[0],feature.shape[1]*4,128,128))
    b = np.zeros((feature.shape[0],feature.shape[1]*4,128,128))
    
    for i in range(feature.shape[0]):
        nv=0
        for v in range(feature.shape[1]):
            if np.amax(feature[i,v,0:128,0:128]) != np.amin(feature[i,v,0:128,0:128]):
            #if np.amax(feature[i,v,0:128,0:128]) * 1e5 >= 1.0:
                a[j,nv,:,:] = feature[i,v,0:128,0:128]
                b[j,nv,:,:] = label[i,v,0:128,0:128]
                nv = nv+1
            if np.amax(feature[i,v,0:128,128:256]) != np.amin(feature[i,v,0:128,128:256]):
            #if np.amax(feature[i,v,0:128,128:256]) * 1e5 >= 1.0:
                a[j,nv,:,:] = feature[i,v,0:128,128:256]
                b[j,nv,:,:] = label[i,v,0:128,128:256]
                nv = nv+1
            if np.amax(feature[i,v,128:256,0:128]) != np.amin(feature[i,v,128:256,0:128]):
            #if np.amax(feature[i,v,128:256,0:128]) * 1e5 >= 1.0:
                a[j,nv,:,:] = feature[i,v,128:256,0:128]
                b[j,nv,:,:] = label[i,v,128:256,0:128]
                nv = nv+1
            if np.amax(feature[i,v,128:256,128:256]) != np.amin(feature[i,v,128:256,128:256]):
            #if np.amax(feature[i,v,128:256,128:256]) * 1e5 >= 1.0:
                a[j,nv,:,:] = feature[i,v,128:256,128:256]
                b[j,nv,:,:] = label[i,v,128:256,128:256]
                nv = nv+1
        if nv > 0:
            j = j+1
            if nv < min_nv:
                min_nv = nv

    print("number of 128x128 images:", j)
    print("Min number of channels:", min_nv)
    
    for i in range(min_nv,feature.shape[1] * 4):
        a = np.delete(a, min_nv, 1)
        b = np.delete(b, min_nv, 1)
       
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
print("Loading Z data")
z = load_data.load_Z_data_all(slevel=0, elevel=50, method='sel')
terrain_height = z[:,0,:,:].copy()
terrain_height = terrain_height[:,np.newaxis,:,:]
print("z.shape =",z.shape)
print("terrain_height.shape =",terrain_height.shape)
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
print("loading W data")
w = load_data.load_W_data_all(slevel=0, elevel=50, method="sel")
print('W.shape =', w.shape)
#
# interpolated the W field to the requested
# heights AGL
#
print("interp W data")
W_VAR = interp_data_concat(w, z)
del w

#
# Should this be normalized using the min and max for entire dataset or normalize based on
# min/max of each time?
#

#for i in range(W_VAR.shape[0] - 1):
#    for v in range(W_VAR.shape[1] - 1):
#        if np.amax(W_VAR[i,v,:,:]) == np.amin(W_VAR[i,v,:,:]):
#            W_VAR[i,v,:,:] == 0
#        else:
#            W_VAR[i,v,:,:] = (W_VAR[i,v,:,:] - np.amin(W_VAR[i,v,:,:])) / (np.amax(W_VAR[i,v,:,:]) - np.amin(W_VAR[i,v,:,:]))
#
print("Doing min/max scaling on W data")
W_VAR = W_VAR - np.amin(W_VAR) / np.amax(W_VAR) - np.amin(W_VAR)
#
#output_data = '/glade/work/hardt/ds612/2000-2013_June-Sept_W_INTERP_AGL_0to7km_at_500m_steps.nc'
output_data = '/glade/work/hardt/ds612/2000-2013_June-Sept_W_INTERP_AGL_0to7km_at_500m_steps2.nc'
WAGL = xr.DataArray(W_VAR, name='W')
encoding={'W': {'zlib': True, '_FillValue': -99.0}}
print("Writing W data to", output_data)
WAGL.to_netcdf(output_data, encoding=encoding)
#
del W_VAR

#
# Load the model feature (QRAIN)
#
#qr = load_data.load_QRAIN_data_oneTime()
print("loading QRAIN data")
qr = load_data.load_QRAIN_data_all(slevel=0, elevel=50, method="sel")
print("qr.shape =",qr.shape)
#
# interpolated the QRAIN field to the requested
# heights AGL 
#
print("interp QRAIN data")
QR_VAR = interp_data_concat(qr, z)
del qr

#
# Should this be normalized using the min and max for entire dataset or normalize based on
# min/max of each time?
#

#for i in range(QR_VAR.shape[0] - 1):
#    for v in range(QR_VAR.shape[1] - 1):
#        if np.amax(QR_VAR[i,v,:,:]) == np.amin(QR_VAR[i,v,:,:]):
#            QR_VAR[i,v,:,:] == 0
#        else:
#            QR_VAR[i,v,:,:] = (QR_VAR[i,v,:,:] - np.amin(QR_VAR[i,v,:,:])) / (np.amax(QR_VAR[i,v,:,:]) - np.amin(QR_VAR[i,v,:,:]))
#
print("Doing min/max scaling of QRAIN data")
QR_VAR = QR_VAR - np.amin(QR_VAR) / np.amax(QR_VAR) - np.amin(QR_VAR)
#
#output_data = '/glade/work/hardt/ds612/2000-2013_June-Sept_QRAIN_INTERP_AGL_0to7km_at_500m_steps.nc'
output_data = '/glade/work/hardt/ds612/2000-2013_June-Sept_QRAIN_INTERP_AGL_0to7km_at_500m_steps2.nc'
QR = xr.DataArray(QR_VAR, name='QRAIN')
encoding={'QRAIN': {'zlib': True, '_FillValue': -99.0}}
print("Wrting output to", output_data)
QR.to_netcdf(output_data, encoding=encoding)
#
del QR_VAR

#
# Load another model feature field (QSNOW)
#
print("loading QSNOW data")
qs = load_data.load_QSNOW_data_all(slevel=0, elevel=50, method="sel")
print(qs.shape)
#
# interpolated the QSNOW field to the requested
# heights AGL
#
print("interp QSNOW data")
QS_VAR = interp_data_concat(qs, z)
del qs

#
# Should this be normalized using the min and max for entire dataset or normalize based on
# min/max of each time?
#

#for i in range(QS_VAR.shape[0] - 1):
#    for v in range(QS_VAR.shape[1] - 1):
#        if np.amax(QS_VAR[i,v,:,:]) == np.amin(QS_VAR[i,v,:,:]):
#            QS_VAR[i,v,:,:] == 0
#        else:
#            QS_VAR[i,v,:,:] = (QS_VAR[i,v,:,:] - np.amin(QS_VAR[i,v,:,:])) / (np.amax(QS_VAR[i,v,:,:]) - np.amin(QS_VAR[i,v,:,:]))
#
print("Doing min/max scaling of QSNOW data")
QS_VAR = QS_VAR - np.amin(QS_VAR) / np.amax(QS_VAR) - np.amin(QS_VAR)
#
#output_data = '/glade/work/hardt/ds612/2000-2013_June-Sept_QSNOW_INTERP_AGL_0to7km_at_500m_steps.nc'
output_data = '/glade/work/hardt/ds612/2000-2013_June-Sept_QSNOW_INTERP_AGL_0to7km_at_500m_steps2.nc'
QS = xr.DataArray(QS_VAR, name='QSNOW')
encoding={'QSNOW': {'zlib': True, '_FillValue': -99.0}}
print("Writing output to", output_data)
QS.to_netcdf(output_data, encoding=encoding)
#
del QS_VAR
