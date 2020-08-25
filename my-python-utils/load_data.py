#!/usr/bin/env python

import xarray as xr
import os

model_main_path = "/glade/collections/rda/data/ds612.0"

def load_refl_data_3yrs():
    data = xr.open_mfdataset(os.path.join(model_main_path, "CTRLradrefl/REFL/wrf2d_d01_CTRL_REFL_10CM_201*07-201*09.nc"), combine="by_coords")
    x = data.REFL_10CM[::3,350:350+256,650:650+256].values

    return x


def load_W_data_3yrs(level=20, method="sel"):
    ds = xr.open_mfdataset(os.path.join(model_main_path, "CTRL3D/201*/wrf3d_d01_CTRL_W_201?0[789]*.nc"), combine="by_coords")
    if method=="sel":
        y = ds.W[:,level,350:350+256,650:650+256].values
    elif method=="max":
        y = ds.W[:,:,350:350+256,650:650+256].values.max(axis=1)
    elif method=="mean":
        y = ds.W[:,:,350:350+256,650:650+256].values.mean(axis=1)

    return y

def load_refl_data_all():
    data = xr.open_mfdataset(os.path.join(model_main_path, "CTRLradrefl/REFL/wrf2d_d01_CTRL_REFL_10CM_20*07-20*09.nc"), combine="by_coords")
    x = data.REFL_10CM[::3,350:350+256,650:650+256].values

    return x


def load_W_data_all(level=20, method="sel"):
    ds = xr.open_mfdataset(os.path.join(model_main_path, "CTRL3D/20*/wrf3d_d01_CTRL_W_20??0[789]*.nc"), combine="by_coords")
    if method=="sel":
        y = ds.W[:,level,350:350+256,650:650+256].values
    elif method=="max":
        y = ds.W[:,:,350:350+256,650:650+256].values.max(axis=1)
    elif method=="mean":
        y = ds.W[:,:,350:350+256,650:650+256].values.mean(axis=1)

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
