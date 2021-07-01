#!/usr/bin/env python

import xarray as xr
from netCDF4 import Dataset
import numpy as np
import os
import sys

#
# read in the 0, 5, 10, and 15 minute reflectivity data
#
# Separate out the test data so no shuffling is done
# 1999-2015 (17 years of data)
# 
# training data: 1999-2009 (images: 11years * 1224 = 13,464)
# validation data: 2010-2014 (images: 5years * 1224 = 6,120)
# test data: 2015 (images: 1,224)
#
# Reflectivity
# 1224 images per year
# 
# scale the data using the min and 99.9 percentile data
#    - refl = (refl - scale_min) / (scale_99p9 - scale_min)
# Write the Reflectivity data for each offset
#    - training data includes the times, and the shuffle sequence
#    - test data includes the times.
#

#global data_input_path = '/glade/work/hardt/data/model2/v1'
global data_input_path = '/glade/campaign/ral/aap/hardt/STEP/AI/model2/v1'

global data_output_path = '/glade/work/hardt/data/model2/v2'
global s

def process_refl(files, offset, outputFilename, ComputeShuffle=True, DoShuffle=True):
    
    chdir(data_input_path)
    
    ds = xr.open_mfdataset(files, combine="nested", concat_dim='Time')
    
    #
    # read in the 3D reflectivity data
    # only get 10 of the levels 3:30:3
    #
    refl = ds.REFL_10CM[:,3:30:3,:,:].values
    refl_t = ds.XTIME.values
    
    #
    # move the channels from position 1 to position 3
    # goes from [time,channel,height,width] to [time, height, width, channel]
    # which is the default for Conv2D.
    #
    refl = np.moveaxis(refl, 1, 3)

    #
    # create the random shuffle indexes
    #
    if ComputeShuffle:
        print("Defining shuffle sequence")
        s = np.arange(refl.shape[0])
        np.random.shuffle(s)
    
    #
    # Shuffle the data
    #
    if DoShuffle:
        refl = refl[s]
        refl_t = refl_t[s]
    
    #
    # get scaling values using all 10 levels
    #
    scale_min = np.amin(refl)
    scale_99p9 = np.percentile(refl, 99.9)
    
    #
    # Scale the data
    #
    print('refl.shape:',refl.shape)
    print("Doing min/max scaling")
    refl = (refl - scale_min) / (scale_99p9 - scale_min)

    #
    # Write the refl data
    #
    
    output_data = os.path.join(data_output_path, outputFilename)
    print("Writing REFL_10CM training data to", output_data)

    REFL_XTIME = xr.DataArray(refl_t, name='XTIME')
    REFL_XTIME.to_netcdf(output_data)
    
    if DoShuffle:
        SHUFFLE = xr.DataArray(s, name='shuffle_seq')
        SHUFFLE.to_netcdf(output_data, mode='a')
        del SHUFFLE

    REFL_OUT = xr.DataArray(data=refl, 
                            name='REFL_10CM',
                            dims=['time', 'south_north','west_east', 'bottom_top'],
                            attrs=dict(
                                description='reflectivity',
                                units='dBZ',
                                scale_min=scale_min,
                                scale_99p9=scale_99p9,
                            ),
                          )

    encoding={'REFL_10CM': {'zlib': True, '_FillValue': -99.0}}
    REFL_OUT.to_netcdf(output_data, encoding=encoding, mode='a')
        
    del ds
    del refl
    del refl_t
    del REFL_OUT
    del REFL_XTIME
    
    print('DONE writing REFL data for time offset', offset)
    print()    

def process_W(files, outputFilename, DoShuffle=True):
    
    chdir(data_input_path)
    
    ds = xr.open_mfdataset(files, combine="nested", concat_dim='Time')
    
    #
    # read in the 3D reflectivity data
    # only get 10 of the levels 3:30:3
    #
    W = ds.REFL_10CM[:,3:30:3,:,:].values
    W_t = ds.XTIME.values
    
    #
    # move the channels from position 1 to position 3
    # goes from [time,channel,height,width] to [time, height, width, channel]
    # which is the default for Conv2D.
    #
    W = np.moveaxis(refl, 1, 3)

    #
    # Shuffle the data
    #
    if DoShuffle:
        refl = refl[s]
        refl_t = refl_t[s]

    #
    # get scaling values using all 10 levels
    #
    scale_min = np.amin(W)
    scale_99p9 = np.percentile(W, 99.9)
    
    W = W[:,:,:,:].values.max(axis=3)

    #
    # Writing the data 
    #
     
    output_data = os.path.join(data_output_path, outputFilename)
    print("Writing W data to", output_data)
    
    W_XTIME = xr.DataArray(W_t, name='XTIME')
    W_XTIME.to_netcdf(output_data)
    
    if DoShuffle:
        SHUFFLE = xr.DataArray(s, name='shuffle_seq')
        SHUFFLE.to_netcdf(output_data, mode='a')
        del SHUFFLE

    W_OUT = xr.DataArray(data=W, 
                         name='W',
                         dims=['time','south_north','west_east'],
                         # dims=['time','south_north','west_east','bottom_top'],
                         attrs=dict(
                             description='W',
                             units='scaled',
                             scale_min=scale_min,
                             scale_99p0=scale_99p0,
                         ),
                        )
    encoding={'W': {'zlib': True, '_FillValue': -999.0}}
    W_OUT.to_netcdf(output_data, encoding=encoding, mode='a')

    del ds
    del W
    del W_t
    del W_OUT
    del W_XTIME
    
    print('DONE writing W data')
    print()
        
def main():
    
    #
    # PROCESSING reflectivity
    #
    
    refl_offsets = ['00', '05', '10', '15']

    for offset in refl_offsets:
    
        training_files = ['model2-v1_1999_' + offset + 'minuteAfterHour_3D_refl.nc', 
                          'model2-v1_2000_' + offset + 'minuteAfterHour_3D_refl.nc', 
                          'model2-v1_2001_' + offset + 'minuteAfterHour_3D_refl.nc', 
                          'model2-v1_2002_' + offset + 'minuteAfterHour_3D_refl.nc', 
                          'model2-v1_2003_' + offset + 'minuteAfterHour_3D_refl.nc', 
                          'model2-v1_2004_' + offset + 'minuteAfterHour_3D_refl.nc', 
                          'model2-v1_2005_' + offset + 'minuteAfterHour_3D_refl.nc', 
                          'model2-v1_2006_' + offset + 'minuteAfterHour_3D_refl.nc', 
                          'model2-v1_2007_' + offset + 'minuteAfterHour_3D_refl.nc', 
                          'model2-v1_2008_' + offset + 'minuteAfterHour_3D_refl.nc', 
                          'model2-v1_2009_' + offset + 'minuteAfterHour_3D_refl.nc']

        validation_files = ['model2-v1_2010_' + offset + 'minuteAfterHour_3D_refl.nc', 
                            'model2-v1_2011_' + offset + 'minuteAfterHour_3D_refl.nc', 
                            'model2-v1_2012_' + offset + 'minuteAfterHour_3D_refl.nc', 
                            'model2-v1_2013_' + offset + 'minuteAfterHour_3D_refl.nc', 
                            'model2-v1_2014_' + offset + 'minuteAfterHour_3D_refl.nc']

        test_files = ['model2-v1_2015_' + offset + 'minuteAfterHour_3D_refl.nc']

        #
        # PROCESSING training data.
        #

        print("Processing refl training files for offset", offset)
        output_file_name = 'model2-v2_1999-2009_' + offset + 'minuteAfterHour_3D_refl-training.nc'
        
        #
        # Only compute the Shuffle sequence once and use it for all the data except the test data
        #
        
        if offset == '00':
            process_refl(training_files, offset, output_file_name, True, True)
        else:
            process_refl(training_files, offset, output_file_name, False, True)
        
        #
        # PROCESSING validation data.
        #
        
        print("Processing refl validation files for offset", offset)   
        output_file_name = 'model2-v2_2010-2014_' + offset + 'minuteAfterHour_3D_refl-validation.nc'
        process_refl(validation_files, offset, output_file_name, False,True)
                
        #
        # PROCESSING test data which does not get shuffled.
        #

        print("Processing refl test files for offset", offset)
        output_file_name = 'model2-v2_2015_' + offset + 'minuteAfterHour_3D_refl-test.nc'
        process_refl(test_files, offset, output_file_name, False, False)
        
    #
    # PROCESSING W
    #
    
    training_files = ['model2-v1_1999_3D_W.nc', 
                      'model2-v1_2000_3D_W.nc', 
                      'model2-v1_2001_3D_W.nc', 
                      'model2-v1_2002_3D_W.nc', 
                      'model2-v1_2003_3D_W.nc', 
                      'model2-v1_2004_3D_W.nc', 
                      'model2-v1_2005_3D_W.nc', 
                      'model2-v1_2006_3D_W.nc', 
                      'model2-v1_2007_3D_W.nc', 
                      'model2-v1_2008_3D_W.nc', 
                      'model2-v1_2009_3D_W.nc']

    validation_files = ['model2-v1_2010_3D_W.nc', 
                        'model2-v1_2011_3D_W.nc', 
                        'model2-v1_2012_3D_W.nc', 
                        'model2-v1_2013_3D_W.nc', 
                        'model2-v1_2014_3D_W.nc']
    
    test_files = ['model2-v1_2015_3D_W.nc']
        
    #
    # PROCESSING training data.
    #

    print("Processing W training files.")
    output_file_name = 'model2-v2_1999-2009_composite_W-training.nc'
    process_W(training_files, offset, output_file_name, True)
        
    #
    # PROCESSING validation data.
    #

    print("Processing W validation files.")
    output_file_name = 'model2-v2_2010-2014_composite_W-validation.nc'
    process_W(validation_files, offset, output_file_name, True)
    
    #
    # PROCESSING test data which does not get shuffled.
    #
    
    print("Processing W test files.")
    output_file_name = 'model2-v2_2015_composite_W-test.nc'
    process_W(test_files, offset, output_file_name, False)
        
        
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
