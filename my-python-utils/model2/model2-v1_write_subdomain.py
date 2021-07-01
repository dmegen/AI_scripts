#!/usr/bin/env python

import xarray as xr
from netCDF4 import Dataset
import numpy as np
import os
import sys

refl_yoffset = 256
refl_xoffset = 185

Field_yoffset = refl_yoffset + 130
Field_xoffset = refl_xoffset + 560

years = ['2014', '2015']

# W, QVAPOR, U, V
fields = ['QVAPOR', 'U', 'V']

input_W_data_path = '/glade/scratch/hardt/step/wrf3d'
output_data_path = '/glade/work/hardt/data/model2/v1'

for year in years:
    
    for field in fields:
        
        print("Processing year", year, "and field", field)

        ds = xr.open_mfdataset(os.path.join(input_W_data_path, year, "wrf3d_d01*"), combine="nested", concat_dim='Time')
        t = ds.XTIME.values
        print('Loading', field, 'data.')
        
        if field == 'W':
            VAR = ds.W[:,:,Field_yoffset:Field_yoffset+256,Field_xoffset:Field_xoffset+256].values
        elif field =='QVAPOR':
            VAR = ds.QVAPOR[:,:,Field_yoffset:Field_yoffset+256,Field_xoffset:Field_xoffset+256].values
        elif field =='U':
            VAR = ds.U[:,:,Field_yoffset:Field_yoffset+256,Field_xoffset:Field_xoffset+256].values
        elif field =='V':
            VAR = ds.V[:,:,Field_yoffset:Field_yoffset+256,Field_xoffset:Field_xoffset+256].values
        else:
            print("Field", field, "not an option.")
            continue
    
        print("VAR.shape:", VAR.shape)
        print("np.amin(VAR):", np.amin(VAR))
        print("np.percentile(VAR, 99.9):", np.percentile(VAR, 99.9))

        output_data = os.path.join(output_data_path, "model2-v1_" + year + '_3D_' + field + '.nc')
        OUT = xr.DataArray(VAR, name=field)
        encoding={field: {'zlib': True, '_FillValue': -99.0}}

        XTIME = xr.DataArray(t, name='XTIME')

        print("Writing", field, "data to", output_data)
        OUT.to_netcdf(output_data, encoding=encoding)
        XTIME.to_netcdf(output_data, mode='a')

        del VAR
        del OUT
        del ds
        del t
        del XTIME
              

        print("Done processing year", year, "and field", field)
        print()
        
