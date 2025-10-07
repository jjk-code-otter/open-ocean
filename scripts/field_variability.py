#   Open Ocean marine data processing
#   Copyright (C) 2025 John Kennedy
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.
from open_ocean import gridder
from open_ocean import interpolation as io
from itertools import product
import json
import xarray as xr
import itertools
import copy
import pandas as pd
import numpy as np
from pathlib import Path
import calendar
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from open_ocean.gridder import Grid
from open_ocean.utils import convert_climatology_to_ocean_areas, convert_dates


def plot_map(ax, ds):
    proj = ccrs.PlateCarree()
    p = ds.sst[0].plot(ax=ax,
                       transform=proj,
                       subplot_kws={'projection': proj},
                       levels=np.arange(0, 1.5, 0.1),
                       )
    p.axes.coastlines()


data_dir = Path(os.getenv("OODIR"))

filepath = data_dir / 'SST_CCI' / 'SST_ANOM_100_19800101_20250630_regridded'
files = filepath.glob('*.nc')
sst_cci = xr.open_mfdataset(files)

ltm = sst_cci.sst_anomaly.groupby("time.month").mean()
ltstdev = sst_cci.sst_anomaly.groupby("time.month").std()

longitude = ltm.lon
latitude = ltm.lat

month = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November',
         'December']

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 9), subplot_kw=dict(projection=ccrs.PlateCarree()))
plt.subplots_adjust(wspace=0, hspace=0)
for i, ax in zip(range(12), axes.ravel()):
    # ax.coastlines(lw=1, color='w')
    x = ax.pcolormesh(longitude, latitude, ltm[i], vmin=-0.5, vmax=5.5, cmap='RdBu_r')
    ax.text(-175, 77, month[i], color='white')
plt.savefig(data_dir / "SST_CCI" / "mean_anomaly.png")

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 9), subplot_kw=dict(projection=ccrs.PlateCarree()))
plt.subplots_adjust(wspace=0, hspace=0)
for i, ax in zip(range(12), axes.ravel()):
    # ax.coastlines(lw=1, color='w')
    x = ax.pcolormesh(longitude, latitude, ltstdev[i], vmin=0.0, vmax=1.5, cmap='inferno')
    ax.text(-175, 77, month[i], color='white')
plt.savefig(data_dir / "SST_CCI" / "stdev_anomaly.png")

five_grid = np.zeros((12, 36, 72))
big_grid = copy.deepcopy(ltstdev.values) # Force load of data
for yy, xx in itertools.product(range(36), range(72)):
    selection = big_grid[:, yy * 5:(yy + 1) * 5, xx * 5:(xx + 1) * 5]
    for m in range(12):
        subsel = selection[m, :, :]
        pick = ~np.isnan(subsel)
        if np.count_nonzero(pick) > 0:
            five_grid[m, yy, xx] = np.mean(subsel[pick])

five_grid = Grid.make_xarray(
    five_grid,
    res=5,
    times=pd.date_range(start='1981-01-15', freq='1MS', periods=12)
)

longitude = five_grid.longitude
latitude = five_grid.latitude

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 9), subplot_kw=dict(projection=ccrs.PlateCarree()))
plt.subplots_adjust(wspace=0, hspace=0)
for i, ax in zip(range(12), axes.ravel()):
    # ax.coastlines(lw=1, color='w')
    x = ax.pcolormesh(longitude, latitude, five_grid.sst[i], vmin=0.0, vmax=1.5, cmap='inferno')
    ax.text(-175, 77, month[i], color='white')
plt.savefig(data_dir / "SST_CCI" / "stdev_anomaly_5x5.png")

five_grid.to_netcdf(data_dir / "SST_CCI" / "SST_Variability_5x5.nc")