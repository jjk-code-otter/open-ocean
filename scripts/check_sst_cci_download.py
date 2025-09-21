import itertools
from pathlib import Path
import os
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import xarray as xa
import cf_xarray
from cartopy.util import add_cyclic_point
import matplotlib.pyplot as plt

data_dir = os.getenv("DATADIR")

output_grid = np.full((365, 180, 360), np.nan)
output_grid_stdev = np.full((365, 180, 360), np.nan)

all_files = []
for i in range(365):
    new_output_file = Path(data_dir) / "SST_CCI_climatology" / f"D{i + 1:03d}.nc"
    df = xa.open_dataset(new_output_file)
    df1 = df['analysed_sst']
    print(new_output_file)
    for xx, yy in itertools.product(range(360), range(180)):
        selection = df1.data[
                    0,
                    yy * 20:(yy + 1) * 20,
                    xx * 20:(xx + 1) * 20
                    ]
        selection = selection[~np.isnan(selection)]
        output_grid[i, yy, xx] = np.mean(selection)

    df1 = df['analysed_sst_std']
    print(new_output_file)
    for xx, yy in itertools.product(range(360), range(180)):
        selection = df1.data[
                    0,
                    yy * 20:(yy + 1) * 20,
                    xx * 20:(xx + 1) * 20
                    ]
        selection = selection[~np.isnan(selection)]
        output_grid_stdev[i, yy, xx] = np.sqrt(np.mean(selection)**2 + 1.1**2)

# Mean
variable = "sst"
times = pd.date_range(start=f'1981-01-01', freq='1D', periods=365)
latitudes = np.mean(df1.lat.data.reshape(-1, 20), axis=1)
longitudes = np.mean(df1.lon.data.reshape(-1, 20), axis=1)

ds = xa.Dataset({
    variable: xa.DataArray(
        data=output_grid,
        dims=['time', 'latitude', 'longitude'],
        coords={'time': times, 'latitude': latitudes, 'longitude': longitudes},
        attrs={'long_name': 'sea-surface temperature', 'units': 'K'}
    )
},
    attrs={'project': 'open ocean'}
)

ds.to_netcdf(Path(data_dir) / "SST_CCI_climatology" / "SST_1x1_daily.nc")

# Standard deviation
variable = "sst"
times = pd.date_range(start=f'1981-01-01', freq='1D', periods=365)
latitudes = np.mean(df1.lat.data.reshape(-1, 20), axis=1)
longitudes = np.mean(df1.lon.data.reshape(-1, 20), axis=1)

ds = xa.Dataset({
    variable: xa.DataArray(
        data=output_grid_stdev,
        dims=['time', 'latitude', 'longitude'],
        coords={'time': times, 'latitude': latitudes, 'longitude': longitudes},
        attrs={'long_name': 'sea-surface temperature standard deviation', 'units': 'K'}
    )
},
    attrs={'project': 'open ocean'}
)

ds.to_netcdf(Path(data_dir) / "SST_CCI_climatology" / "SST_stdev_1x1_daily.nc")

# Standard deviation single field
variable = "sst"
times = pd.date_range(start=f'1981-01-01', freq='1D', periods=1)
latitudes = np.mean(df1.lat.data.reshape(-1, 20), axis=1)
longitudes = np.mean(df1.lon.data.reshape(-1, 20), axis=1)

output_grid_stdev_single = np.mean(output_grid_stdev, axis=0, keepdims=True)

ds = xa.Dataset({
    variable: xa.DataArray(
        data=output_grid_stdev_single,
        dims=['time', 'latitude', 'longitude'],
        coords={'time': times, 'latitude': latitudes, 'longitude': longitudes},
        attrs={'long_name': 'sea-surface temperature standard deviation', 'units': 'K'}
    )
},
    attrs={'project': 'open ocean'}
)

ds.to_netcdf(Path(data_dir) / "SST_CCI_climatology" / "SST_stdev_1x1_single_field.nc")
