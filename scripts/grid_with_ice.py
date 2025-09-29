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
import itertools

from open_ocean import gridder
from open_ocean import interpolation as io
from itertools import product
import json
import xarray as xr
import netCDF4
import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime, date
import matplotlib.pyplot as plt

from open_ocean.utils import convert_climatology_to_ocean_areas


def fix_ice_array(year, month, ice_ym):
    """Coerce the ice array into the necessary shape"""
    ice_ym = ice_ym.sic.values
    ice_ym = np.flip(ice_ym, 0) # latitudes are specified upside down
    ice_ym = np.roll(ice_ym, 180, 1) # longitudes start at 0 not -180

    latitude = np.linspace(-89.5, 89.5, 180)
    latitude = np.reshape(latitude, (180, 1))
    latitude = np.repeat(latitude, 360, 1)

    longitude = np.linspace(-179.5, 179.5, 360)
    longitude = np.reshape(longitude, (1, 360))
    longitude = np.repeat(longitude, 180, 0)

    selection = (ice_ym >= 0.8)

    ice_ym = ice_ym[selection]
    longitude = longitude[selection]
    latitude = latitude[selection]

    months = np.array([month for _ in range(len(ice_ym))])
    days = np.array([14 for _ in range(len(ice_ym))])
    dates = convert_dates(months, days)

    # id = np.random.uniform(0, 100, len(ice_ym))
    # id = [f'{x:.3f}' for x in id]
    id = ['ICE' for _ in range(len(ice_ym))]

    values = [273.15-1.8 for _ in range(len(ice_ym))]
    type = [99 for _ in range(len(ice_ym))]

    # ice5 = np.zeros((1, 36, 72))
    #
    # for xx, yy in itertools.product(range(72), range(36)):
    #     selection = ice_ym[0, yy*5:(yy+1)*5, xx*5:(xx+1)*5]
    #     ice5[0, yy, xx] = np.mean(selection[~np.isnan(selection)])

    return longitude, latitude, dates, id, values, type


def convert_dates(months, days):
    return [datetime(2020, months[i], days[i]) for i in range(len(months))]


def grid_selection(year, month, iquam, selection, climatology, sampling_unc, ice, constant=None):
    id = iquam.platform_id.values[selection]
    type = iquam.platform_type.values[selection]
    lats = iquam.lat.values[selection]
    lons = iquam.lon.values[selection]
    values = iquam.sst.values[selection]

    # Convert dates
    dates = convert_dates(
        iquam.month.values[selection].astype(int),
        iquam.day.values[selection].astype(int)
    )

    # Add ice
    ice_lon, ice_lat, ice_dates, ice_id, ice_values, ice_type = fix_ice_array(year, month, ice)

    id = np.concatenate([id, ice_id])
    type = np.concatenate([type, ice_type])
    lats = np.concatenate([lats, ice_lat])
    lons = np.concatenate([lons, ice_lon])
    values = np.concatenate([values, ice_values])

    dates = dates + ice_dates

    # Grid up the data
    grid = gridder.Grid(2020, 10, id, lats, lons, dates, values, type, climatology)
    grid.add_sampling_uncertainties(sampling_unc)
    grid.do_1x1_gridding()
    grid.do_one_step_5x5_gridding()
    grid.calculate_covariance(constant=constant)

    return grid


if __name__ == "__main__":
    data_dir = Path(os.getenv("OODIR"))
    coder = xr.coders.CFDatetimeCoder(time_unit="s")

    ts = []
    ts_unc = []
    time = []

    with open('regions.json', 'r') as f:
        regions = json.load(f)

    climatology = xr.open_dataset(data_dir / "SST_CCI_climatology" / "SST_1x1_daily.nc")
    areas = convert_climatology_to_ocean_areas(climatology)
    sampling_unc = xr.open_dataset(data_dir / "IQUAM" / "OutputData" / "sampling_uncertainty.nc")
    ice = xr.open_dataset(data_dir / "IQUAM" / "InputData" / "HadISST.2.2.0.0_sea_ice_concentration.nc", engine='netcdf4')

    n_time = (2025 - 1981 + 1) * 12

    all_data = np.zeros((n_time, 36, 72)) + np.nan
    all_nobs = np.zeros((n_time, 36, 72))
    all_unc = np.zeros((n_time, 36, 72)) + np.nan
    all_interpolate = np.zeros((n_time, 36, 72)) + np.nan

    interp_data = np.zeros((n_time, 36, 72)) + np.nan
    interp_unc = np.zeros((n_time, 36, 72)) + np.nan

    region_names = [key for key in regions.keys()]
    component_names = [
        "all", "all_unc",
        "ship", "ship_unc",
        "drifter", "drifter_unc",
        "argo", "argo_unc",
        "interp", "interp_unc"
    ]

    mux = pd.MultiIndex.from_product([component_names, region_names])
    time_series = pd.DataFrame(columns=mux)

    count = -1

    for year, month in product(range(1981, 2026), range(1, 13)):
        file = data_dir / 'IQUAM' / f'{year}{month:02d}-STAR-L2i_GHRSST-SST-iQuam-V2.10-v01.0-fv01.0.nc'

        if not (file.exists()):
            continue

        iceym = ice.sel(time=f"{year}-{month:02d}", method="nearest")

        iquam = xr.open_dataset(file, decode_timedelta=coder)

        # Select only high quality observations
        quality = iquam.quality_level.values
        pt = iquam.platform_type.values
        selection = (quality >= 4)

        count += 1

        row = []

        grid = grid_selection(year, month, iquam, selection, climatology, sampling_unc, iceym)
        for key, entry in regions.items():
            gmsst, gmsst_unc = grid.calculate_area_average_with_covariance(
                areas=areas, lat_range=entry["lat_range"], lon_range=entry["lon_range"]
            )
            row.append(gmsst)
            row.append(gmsst_unc)
            print(f"{key} {year} {month:02d}: {gmsst:.3f} ± {gmsst_unc:.3f}")

        kernel = io.Kernel(0.6, 1300.0, 1.5)
        interp = io.GPInterpolator(grid, kernel)
        interp.make_covariance(constant=0.2)
        interpolated_grid = interp.do_interpolation()
        interpolated_grid.data5[np.isnan(sampling_unc.sst.values[0:1, :, :])] = np.nan

        all_data[count, :, :] = grid.data5[0, :, :]
        all_interpolate[count, :, :] = interpolated_grid.data5[0, :, :]
        all_nobs[count, :, :] = grid.numobs5[0, :, :]
        all_unc[count, :, :] = grid.unc5[0, :, :]

        # Plot some progress plots
        grid.plot_map_1x1(filename=data_dir / "IQUAM" / "Figures" / f"ice_one_deg_{year}{month:02d}.png")
        grid.plot_map_5x5(filename=data_dir / "IQUAM" / "Figures" / f"ice_five_deg_{year}{month:02d}.png")
        interpolated_grid.plot_map_5x5(
            filename=data_dir / "IQUAM" / "Figures" / f"ice_five_deg_interp_{year}{month:02d}.png")
        grid.plot_map_unc_5x5(filename=data_dir / "IQUAM" / "Figures" / f"ice_unc_{year}{month:02d}.png")
        interpolated_grid.plot_map_unc_5x5(
            filename=data_dir / "IQUAM" / "Figures" / f"ice_unc_interp_{year}{month:02d}.png")

        # difference = grid - interpolated_grid
        # difference.plot_map_5x5()

        # Calculate the area average for the grid
        ts.append(gmsst)
        ts_unc.append(gmsst_unc)
        time.append(year + (month - 1) / 12.)

        # time_series.loc[count] = row
        # time_series.to_csv(data_dir / "IQUAM" / "OutputData" / "timeseries_with_uncertainty.csv")

    # avinterp = time_series['interp']
    # avinterp_unc = time_series['interp_unc']
    # plt.fill_between(
    #     time, avinterp['Global'] + 2 * avinterp_unc['Global'], avinterp['Global'] - 2 * avinterp_unc['Global'],
    #     label="Interpolated", color="red", alpha=0.5
    # )
    #
    # plt.xlim(1980, 2027)
    # plt.ylim(-0.5, 0.85)
    #
    # plt.legend()
    # plt.savefig(data_dir / "IQUAM" / "Figures" / "timeseries_with_uncertainty.png")
    #
    # time_series.to_csv(data_dir / "IQUAM" / "OutputData" / "timeseries_with_uncertainty.csv")
    #
    # # Transfer the data to xarray DataArrays and write out
    # all_data = all_data[0:count + 1, :, :]
    # all_interpolate = all_interpolate[0:count + 1, :, :]
    # all_unc = all_unc[0:count + 1, :, :]
    # all_nobs = all_nobs[0:count + 1, :, :]
    #
    # interp_data = interp_data[0:count + 1, :, :]
    # interp_unc = interp_unc[0:count + 1, :, :]
    #
    # date_range = pd.date_range(start=f'1981-09-01', freq='1MS', periods=count + 1)
    #
    # oo_anomalies = gridder.Grid.make_xarray(all_data, res=5, times=date_range)
    # oo_interpolated = gridder.Grid.make_xarray(all_interpolate, res=5, times=date_range)
    # oo_uncertainty = gridder.Grid.make_xarray(all_unc, res=5, times=date_range)
    # oo_numobs = gridder.Grid.make_xarray(all_nobs, res=5, times=date_range)
    #
    # oo_anomalies.to_netcdf(data_dir / "IQUAM" / "oo_anomalies.nc")
    # oo_interpolated.to_netcdf(data_dir / "IQUAM" / "oo_interpolated.nc")
    # oo_uncertainty.to_netcdf(data_dir / "IQUAM" / "oo_uncertainty.nc")
    # oo_numobs.to_netcdf(data_dir / "IQUAM" / "oo_numobs.nc")
    #
    # oo_anomalies = gridder.Grid.make_xarray(interp_data, res=5, times=date_range)
    # oo_uncertainty = gridder.Grid.make_xarray(interp_unc, res=5, times=date_range)
    #
    # oo_anomalies.to_netcdf(data_dir / "IQUAM" / "oo_anomalies_interp_adjusted.nc")
    # oo_uncertainty.to_netcdf(data_dir / "IQUAM" / "oo_uncertainty_interp_adjusted.nc")
