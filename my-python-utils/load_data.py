#!/usr/bin/env python

import xarray as xr
import os

model_main_path = "/glade/collections/rda/data/ds612.0"

def load_refl_data_oneTime():
    ds = xr.open_mfdataset(os.path.join(model_main_path, "CTRLradrefl/REFL/wrf2d_d01_CTRL_REFL_10CM_201301-201303.nc"), combine="by_coords")
    # model W data is every 3 hours, reflectivity is every hour (refl at lowest model level)
    x = ds.REFL_10CM[::3,350:350+256,650:650+256].values
    
    return x

def load_Z_data_oneTime():
    ds = xr.open_mfdataset(os.path.join(model_main_path, "CTRL3D/2013/wrf3d_d01_CTRL_Z_20130331.nc"), combine="by_coords")
    y = ds.Z[:,0:50,350:350+256,650:650+256].values
    
    return y

def load_W_data_oneTime():
    ds = xr.open_mfdataset(os.path.join(model_main_path, "CTRL3D/2013/wrf3d_d01_CTRL_W_20130331.nc"), combine="by_coords")
    y = ds.W[:,0:50,350:350+256,650:650+256].values
    
    return y

def load_QRAIN_data_oneTime():
    ds = xr.open_mfdataset(os.path.join(model_main_path, "CTRL3D/2013/wrf3d_d01_CTRL_QRAIN_201303.nc"), combine="by_coords")
    y = ds.QRAIN[240:248,0:50,350:350+256,650:650+256].values
    
    return y

def load_QSNOW_data_oneTime():
    ds = xr.open_mfdataset(os.path.join(model_main_path, "CTRL3D/2013/wrf3d_d01_CTRL_QSNOW_20130331.nc"), combine="by_coords")
    y = ds.QSNOW[:,0:50,350:350+256,650:650+256].values
    
    return y

def load_refl_data_3yrs():
    data = xr.open_mfdataset(os.path.join(model_main_path, "CTRLradrefl/REFL/wrf2d_d01_CTRL_REFL_10CM_201*07-201*09.nc"), combine="by_coords")
    x = data.REFL_10CM[::3,350:350+256,650:650+256].values

    return x

def load_W_data_3yrs(slevel=20, elevel=20, method="sel"):
    ds = xr.open_mfdataset(os.path.join(model_main_path, "CTRL3D/201*/wrf3d_d01_CTRL_W_201?0[789]*.nc"), combine="by_coords")
    if method=="sel":
        y = ds.W[:,slevel:elevel,350:350+256,650:650+256].values
    elif method=="max":
        y = ds.W[:,:,350:350+256,650:650+256].values.max(axis=1)
    elif method=="mean":
        y = ds.W[:,:,350:350+256,650:650+256].values.mean(axis=1)

    return y



def load_refl_data_all():
    data = xr.open_mfdataset(os.path.join(model_main_path, "CTRLradrefl/REFL/wrf2d_d01_CTRL_REFL_10CM_20*07-20*09.nc"), combine="by_coords")
    x = data.REFL_10CM[::3,350:350+256,650:650+256].values

    return x

def load_W_data_all(slevel=20, elevel=20, method="sel"):
    ds = xr.open_mfdataset(os.path.join(model_main_path, "CTRL3D/20*/wrf3d_d01_CTRL_W_20??0[789]*.nc"), combine="by_coords")
    if method=="sel":
        y = ds.W[:,slevel:elevel,350:350+256,650:650+256].values
    elif method=="max":
        y = ds.W[:,:,350:350+256,650:650+256].values.max(axis=1)
    elif method=="mean":
        y = ds.W[:,:,350:350+256,650:650+256].values.mean(axis=1)

    return y

def load_Z_data_all(slevel=20, elevel=20, method="sel"):
    ds = xr.open_mfdataset(os.path.join(model_main_path, "CTRL3D/20*/wrf3d_d01_CTRL_Z_20??0[789]*.nc"), combine="by_coords")
    if method=="sel":
        y = ds.Z[:,slevel:elevel,350:350+256,650:650+256].values
    elif method=="max":
        y = ds.Z[:,:,350:350+256,650:650+256].values.max(axis=1)
    elif method=="mean":
        y = ds.Z[:,:,350:350+256,650:650+256].values.mean(axis=1)

    return y

def load_QRAIN_data_all(slevel=20, elevel=20, method="sel"):
    ds = xr.open_mfdataset(os.path.join(model_main_path, "CTRL3D/20*/wrf3d_d01_CTRL_QRAIN_20??0[789]*.nc"), combine="by_coords")
    if method=="sel":
        y = ds.QRAIN[:,slevel:elevel,350:350+256,650:650+256].values
    elif method=="max":
        y = ds.QRAIN[:,:,350:350+256,650:650+256].values.max(axis=1)
    elif method=="mean":
        y = ds.QRAIN[:,:,350:350+256,650:650+256].values.mean(axis=1)

    return y

def load_QSNOW_data_all(slevel=20, elevel=20, method="sel"):
    ds = xr.open_mfdataset(os.path.join(model_main_path, "CTRL3D/20*/wrf3d_d01_CTRL_QSNOW_20??0[789]*.nc"), combine="by_coords")
    if method=="sel":
        y = ds.QSNOW[:,slevel:elevel,350:350+256,650:650+256].values
    elif method=="max":
        y = ds.QSNOW[:,:,350:350+256,650:650+256].values.max(axis=1)
    elif method=="mean":
        y = ds.QSNOW[:,:,350:350+256,650:650+256].values.mean(axis=1)

    return y





def load_data():
    ntimes = 1000
    data = xr.open_dataset(os.path.join(model_main_path, "CTRLradrefl/REFL/wrf2d_d01_CTRL_REFL_10CM_201007-201009.nc"), combine="by_coords")
    x = data.REFL_10CM[:ntimes*3:3,350:350+256,650:650+256].values

    ds = xr.open_mfdataset(os.path.join(model_main_path, "/CTRL3D/2010/wrf3d_d01_CTRL_W_20100[789]*"), combine="by_coords")
    y = ds.W[:ntimes,20,350:350+256,650:650+256].values

    return x,y


def medium_data(level=20, method="sel"):
    data = xr.open_mfdataset(os.path.join(model_main_path, "CTRLradrefl/REFL/wrf2d_d01_CTRL_REFL_10CM_201*07-201*09.nc"), combine="by_coords")
    x = data.REFL_10CM[::3,350:350+256,650:650+256].values

    ds = xr.open_mfdataset(os.path.join(model_main_path, "CTRL3D/201*/wrf3d_d01_CTRL_W_201?0[789]*.nc"), combine="by_coords")
    if method=="sel":
        y = ds.W[:,level,350:350+256,650:650+256].values
    elif method=="max":
        y = ds.W[:,:,350:350+256,650:650+256].values.max(axis=1)
    elif method=="mean":
        y = ds.W[:,:,350:350+256,650:650+256].values.mean(axis=1)


    return x,y


def large_data(level=20):
    data = xr.open_mfdataset(os.path.join(model_main_path, "CTRLradrefl/REFL/wrf2d_d01_CTRL_REFL_10CM_20*07-20*09.nc"), combine="by_coords")
    x = data.REFL_10CM[::3,350:350+256,650:650+256].values

    try:
        ds = xr.opendataset("preloaded/wrf_conus_Jul-Sept_W.nc")
    except:
        ds = xr.open_mfdataset(os.path.join(model_main_path, "CTRL3D/20*/wrf3d_d01_CTRL_W_20??0[789]*.nc"), combine="by_coords")
    y = ds.W[:,level,350:350+256,650:650+256].values

    return x,y
