#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
from netCDF4 import Dataset
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import load_data
import wrf


# In[2]:

#
# Reading in the 2012-2015 model2 data (only 2012-2015 available initially)
# inputs:
#  latlon: /glade/work/kyoko/CONUS_HYDRO/PhaseII/src/production04km/wrf04km_refl_clipped_coord.nc
#  refl:   /glade/scratch/gutmann/step/20*/uncompressed/* 
#  W:      /glade/scratch/gutmann/step/wrf3d/20*/uncompressed/*
#
# subsectioning to 256x256 domain
# Offsets are different for refl and W since refl is subdomain of W
#  refl_yoffset = 256
#  refl_xoffset = 185
#  W_yoffset = refl_yoffset + 130
#  W_xoffset = refl_xoffset + 560
#
# writing a separate lat lon file
# breaking the reflectivity into 0, 5, 10, and 15 minutes after the hour files
# adding the times to each output
# no other changes are made to the data
#
# output files are as follows:
#  /glade/work/hardt/data/model2/model2_latlon_coord.nc
#  /glade/work/hardt/data/model2/model2_2012-2015_00minuteAfterHour_3D_refl.nc
#  /glade/work/hardt/data/model2/model2_2012-2015_05minuteAfterHour_3D_refl.nc
#  /glade/work/hardt/data/model2/model2_2012-2015_10minuteAfterHour_3D_refl.nc
#  /glade/work/hardt/data/model2/model2_2012-2015_15minuteAfterHour_3D_refl.nc
#  /glade/work/hardt/data/model2/model2_2012-2015_3D_W.nc
#

refl_yoffset = 256
refl_xoffset = 185


# In[3]:


#
# This is the latlon arrays that go with the refl data
#
#latlonFile = '/glade/work/kyoko/CONUS_HYDRO/PhaseII/src/production04km/wrf04km_refl_clipped_coord.nc'
#latlonFile - '/glade/work/hardt/step/wrf5nm/wrf04km_refl_clipped_coord.nc'
#latlon_ds = xr.open_mfdataset(latlonFile, combine="nested", concat_dim='TIME')


# In[4]:


#output_data = '/glade/work/hardt/data/model2/model2-v1_latlon_coord.nc'
#print("Writing latlon data to", output_data)
#
#latlon_ds.XLAT[:,refl_yoffset:refl_yoffset+256,refl_xoffset:refl_xoffset+256].to_netcdf(output_data)
#latlon_ds.XLONG[:,refl_yoffset:refl_yoffset+256,refl_xoffset:refl_xoffset+256].to_netcdf(output_data, mode='a')
#
#del latlon_ds
#print('DONE writing output data.')


# In[5]:


#
# Start with the reflectivity data which is every 5 minutes
# Do the 5 minute after the hour first
# need every 3 hours to match up with the W field
# set W to -99.0 where reflectivity is < some dbz.
# scale relf and W using min/max scaling
# convert to AGL
# write the data
#

refl_data_main_path = '/glade/scratch/hardt/step/'

#
# for testing only get one file
#
#refl_ds = xr.open_mfdataset(os.path.join(refl_data_main_path, "2014/uncompressed/wrf5mn_d01_2014-09-30_00:00:00"), combine="nested", concat_dim='Time')

#
# grab all available times 
# currently Ethan has only grabbed 2012-2015
#
#refl_ds = xr.open_mfdataset(os.path.join(refl_data_main_path, "20*/uncompressed/*"), combine="nested", concat_dim='Time')
refl_ds = xr.open_mfdataset(os.path.join(refl_data_main_path, "2009/wrf5mn_d01*"), combine="nested", concat_dim='Time')


# In[6]:


# 0 time offset
#
# start locations determined in input_models_domain_compare.ipynb
#
# no offset
print("Loading REFL_10CM data.")
refl = refl_ds.REFL_10CM[0::36,:,refl_yoffset:refl_yoffset+256,refl_xoffset:refl_xoffset+256].values
refl_t = refl_ds.XTIME[0::36].values
#
print(refl.shape)
print('np.amin(refl):',np.amin(refl))
print('np.percentile(refl, 99.99):', np.percentile(refl, 99.99))
#
# Write netcdf output. 
# Would like to re-write this using the netcdf4 module.
# adding in attributes so I can store the scaling information
# and also add in the XLONG, XLAT fields.
#
#output_data = '/glade/work/hardt/data/model2/model2-v1_2012-2015_00minuteAfterHour_3D_refl.nc'
output_data = '/glade/work/hardt/data/model2/model2-v1_2009_00minuteAfterHour_3D_refl.nc'
#
REFL_OUT = xr.DataArray(refl, name='REFL_10CM')
#REFL_OUT = xr.DataArray(data=refl, 
#                        name='REFL_10CM',
#                        dims=['time', 'bottom_top', 'south_north','west_east'],
#                        attrs=dict(
#                            description='reflectivity',
#                            units='dBZ',
#                        ),
#                      )

encoding={'REFL_10CM': {'zlib': True, '_FillValue': -99.0}}
#
REFL_XTIME = xr.DataArray(refl_t, name='XTIME')
#REFL_XTIME = xr.DataArray(data=refl_t, 
#                            name='XTIME',
#                            dims=['time'],
#                          )

#encoding={'XTIME': {'zlib': True, '_FillValue': -99.0}}
#
print("Writing REFL_10CM data to", output_data)
REFL_OUT.to_netcdf(output_data, encoding=encoding)
REFL_XTIME.to_netcdf(output_data, mode='a')
#latlon_ds.XLONG.to_netcdf(output_data, mode='a')
#latlon_ds.XLAT.to_netcdf(output_data, mode='a')
#
del refl
del refl_t
del REFL_OUT
del REFL_XTIME
#
print('DONE writing REFL_OUT.')


# In[7]:


# 5 time offset
#
# start locations determined in input_models_domain_compare.ipynb
#
# no offset
print("Loading REFL_10CM data.")
refl = refl_ds.REFL_10CM[1::36,:,refl_yoffset:refl_yoffset+256,refl_xoffset:refl_xoffset+256].values
refl_t = refl_ds.XTIME[1::36].values
#
print(refl.shape)
print('np.amin(refl):',np.amin(refl))
print('np.percentile(refl, 99.99):', np.percentile(refl, 99.99))
#
# Write netcdf output. 
# Would like to re-write this using the netcdf4 module.
# adding in attributes so I can store the scaling information
# and also add in the XLONG, XLAT fields.
#
#output_data = '/glade/work/hardt/data/model2/model2-v1_2012-2015_05minuteAfterHour_3D_refl.nc'
output_data = '/glade/work/hardt/data/model2/model2-v1_2009_05minuteAfterHour_3D_refl.nc'
#
REFL_OUT = xr.DataArray(refl, name='REFL_10CM')
encoding={'REFL_10CM': {'zlib': True, '_FillValue': -99.0}}
#
REFL_XTIME = xr.DataArray(refl_t, name='XTIME')
#encoding={'XTIME': {'zlib': True, '_FillValue': -99.0}}
#
print("Writing REFL_10CM data to", output_data)
REFL_OUT.to_netcdf(output_data, encoding=encoding)
REFL_XTIME.to_netcdf(output_data, mode='a')
#latlon_ds.XLONG.to_netcdf(output_data, mode='a')
#latlon_ds.XLAT.to_netcdf(output_data, mode='a')
#
del refl
del refl_t
del REFL_OUT
del REFL_XTIME
#del refl_ds
print('DONE writing output data.')


# In[8]:


# 10 time offset
#
# start locations determined in input_models_domain_compare.ipynb
#
# no offset
refl = refl_ds.REFL_10CM[2::36,:,refl_yoffset:refl_yoffset+256,refl_xoffset:refl_xoffset+256].values
refl_t = refl_ds.XTIME[2::36].values
#
print(refl.shape)
print('np.amin(refl):',np.amin(refl))
print('np.percentile(refl, 99.99):', np.percentile(refl, 99.99))
#
# Write netcdf output. 
# Would like to re-write this using the netcdf4 module.
# adding in attributes so I can store the scaling information
# and also add in the XLONG, XLAT fields.
#
#output_data = '/glade/work/hardt/data/model2/model2-v1_2012-2015_10minuteAfterHour_3D_refl.nc'
output_data = '/glade/work/hardt/data/model2/model2-v1_2009_10minuteAfterHour_3D_refl.nc'
#
REFL_OUT = xr.DataArray(refl, name='REFL_10CM')
encoding={'REFL_10CM': {'zlib': True, '_FillValue': -99.0}}
#
REFL_XTIME = xr.DataArray(refl_t, name='XTIME')
#encoding={'XTIME': {'zlib': True, '_FillValue': -99.0}}
#
print("Writing REFL_10CM data to", output_data)
REFL_OUT.to_netcdf(output_data, encoding=encoding)
REFL_XTIME.to_netcdf(output_data, mode='a')
#latlon_ds.XLONG.to_netcdf(output_data, mode='a')
#latlon_ds.XLAT.to_netcdf(output_data, mode='a')
#
del refl
del refl_t
del REFL_OUT
del REFL_XTIME
#del refl_ds
print('DONE writing output data.')


# In[9]:


# 15 time offset
#
# start locations determined in input_models_domain_compare.ipynb
#
# no offset
refl = refl_ds.REFL_10CM[3::36,:,refl_yoffset:refl_yoffset+256,refl_xoffset:refl_xoffset+256].values
refl_t = refl_ds.XTIME[3::36].values
#
print(refl.shape)
print('np.amin(refl):',np.amin(refl))
print('np.percentile(refl, 99.99):', np.percentile(refl, 99.99))
#
# Write netcdf output. 
# Would like to re-write this using the netcdf4 module.
# adding in attributes so I can store the scaling information
# and also add in the XLONG, XLAT fields.
#
#output_data = '/glade/work/hardt/data/model2/model2-v1_2012-2015_15minuteAfterHour_3D_refl.nc'
output_data = '/glade/work/hardt/data/model2/model2-v1_2009_15minuteAfterHour_3D_refl.nc'
#
REFL_OUT = xr.DataArray(refl, name='REFL_10CM')
encoding={'REFL_10CM': {'zlib': True, '_FillValue': -99.0}}
#
REFL_XTIME = xr.DataArray(refl_t, name='XTIME')
#encoding={'XTIME': {'zlib': True, '_FillValue': -99.0}}
#
print("Writing REFL_10CM data to", output_data)
REFL_OUT.to_netcdf(output_data, encoding=encoding)
REFL_XTIME.to_netcdf(output_data, mode='a')
#latlon_ds.XLONG.to_netcdf(output_data, mode='a')
#latlon_ds.XLAT.to_netcdf(output_data, mode='a')
#
del refl
del refl_t
del REFL_OUT
del REFL_XTIME
#del refl_ds
print('DONE writing output data.')


# In[10]:


#
# read in the W data
#

W_data_main_path = '/glade/scratch/gutmann/step/wrf3d/'

#
# for testing only get the one file.
#
#W_ds = xr.open_mfdataset(os.path.join(W_data_main_path, "2014/uncompressed/wrf3d_d01_2014-09-30_*"), combine="nested", concat_dim='Time')

#
# Read in all the data. As of 12/10/2020 Ethan only has 2012-2015
#
#W_ds = xr.open_mfdataset(os.path.join(W_data_main_path, "20*/uncompressed/*"), combine="nested", concat_dim='Time')
W_ds = xr.open_mfdataset(os.path.join(W_data_main_path, "2009/wrf3d_d01*"), combine="nested", concat_dim='Time')
#
W_yoffset = refl_yoffset + 130
W_xoffset = refl_xoffset + 560
print('W_yoffset:',W_yoffset)
print('W_xoffset:',W_xoffset)
#
W_t = W_ds.XTIME.values
#
# start values determined in refl_ds.attrs history
# ncks -O -dwest_east,560,1320 -dsouth_north,130,955
#
print('Loading W data.')
W = W_ds.W[:,:,W_yoffset:W_yoffset+256,W_xoffset:W_xoffset+256].values
#W = W_ds.W[:,:,W_yoffset:W_yoffset+256,W_xoffset:W_xoffset+256].values.max(axis=1)
#
print(W.shape)
print('np.amin(W):',np.amin(W))
print('np.percentile(W, 99.9):',np.percentile(W, 99.9))
#
#output_data = '/glade/work/hardt/data/model2/model2-v1_2012-2015_3D_W.nc'
output_data = '/glade/work/hardt/data/model2/model2-v1_2009_3D_W.nc'
#
W_OUT = xr.DataArray(W, name='W')
encoding={'W': {'zlib': True, '_FillValue': -99.0}}
#
W_XTIME = xr.DataArray(W_t, name='XTIME')
#
print("Writing W data to", output_data)
W_OUT.to_netcdf(output_data, encoding=encoding)
W_XTIME.to_netcdf(output_data, mode='a')
#
del W
del W_OUT
del W_ds
print("Done")

