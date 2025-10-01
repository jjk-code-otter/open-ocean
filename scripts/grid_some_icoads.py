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
import copy
import pandas as pd
import numpy as np
from pathlib import Path
import calendar
import os
from datetime import datetime
import matplotlib.pyplot as plt

from open_ocean.utils import convert_climatology_to_ocean_areas, convert_dates


def grid_selection(df, selection, climatology, sampling_unc, constant=None, separates=False):
    id = df.id.values[selection]
    type = df.pt.values[selection]
    lats = df.lat.values[selection]
    lons = df.lon.values[selection]
    values = df.sst.values[selection] + 273.15
    days = df.day.values[selection]
    months = df.month.values[selection]
    deck = df.dck.values[selection]

    lons[lons > 180.0] = lons[lons > 180.0] - 360.0

    pt_copy = copy.deepcopy(type)

    SHIP = 1
    DRIFT = 2
    MOOR = 3
    ARGO = 5

    pt_copy[:] = SHIP
    pt_copy[type == 7] = DRIFT
    pt_copy[type == 6] = MOOR

    type = pt_copy

    month_lengths = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    if calendar.isleap(df.year.values[0]):
        month_lengths[1] = 29

    valid_days = (days > 0) & (days <= month_lengths[months - 1])

    id = id[valid_days]
    type = type[valid_days]
    lats = lats[valid_days]
    lons = lons[valid_days]
    values = values[valid_days]
    months = months[valid_days]
    days = days[valid_days]
    deck = deck[valid_days]

    # Convert dates
    dates = convert_dates(months.astype(int), days.astype(int))

    # Grid up the data
    grid = gridder.Grid(2020, 10, id, lats, lons, dates, values, type, climatology)
    grid.add_sampling_uncertainties(sampling_unc)
    grid.do_1x1_gridding()
    grid.do_one_step_5x5_gridding()
    bias_cov = grid.calculate_covariance(constant=constant, separates=separates)
    deck_cov = grid.add_correlated_error('deck', deck, 0.2)

    if separates:
        return grid, bias_cov, deck_cov

    return grid


if __name__ == '__main__':
    data_dir = Path(os.getenv("OODIR"))  #

    ts = []
    ts_unc = []
    time = []

    with open('regions.json', 'r') as f:
        regions = json.load(f)

    climatology = xr.open_dataset(data_dir / "SST_CCI_climatology" / "SST_1x1_daily.nc")
    areas = convert_climatology_to_ocean_areas(climatology)
    sampling_unc = xr.open_dataset(data_dir / "IQUAM" / "OutputData" / "sampling_uncertainty.nc")

    n_time = (2025 - 1850 + 1) * 12

    all_data = np.zeros((n_time, 36, 72)) + np.nan
    all_nobs = np.zeros((n_time, 36, 72))
    all_unc = np.zeros((n_time, 36, 72)) + np.nan
    all_interpolate = np.zeros((n_time, 36, 72)) + np.nan

    ship_data = np.zeros((n_time, 36, 72)) + np.nan
    ship_nobs = np.zeros((n_time, 36, 72))
    ship_unc = np.zeros((n_time, 36, 72)) + np.nan

    drifter_data = np.zeros((n_time, 36, 72)) + np.nan
    drifter_nobs = np.zeros((n_time, 36, 72))
    drifter_unc = np.zeros((n_time, 36, 72)) + np.nan

    interp_data = np.zeros((n_time, 36, 72)) + np.nan
    interp_unc = np.zeros((n_time, 36, 72)) + np.nan

    region_names = [key for key in regions.keys()]
    component_names = [
        "all", "all_unc",
        "ship", "ship_unc",
        "drifter", "drifter_unc",
        "interp", "interp_unc"
    ]

    mux = pd.MultiIndex.from_product([component_names, region_names])
    time_series = pd.DataFrame(columns=mux)

    count = -1

    for year, month in product(range(1850, 1858), range(1, 13)):
        print(year, month)

        file = data_dir / "ICOADS" / f"icoads_{year}{month:02d}.csv"

        df = pd.read_csv(file)

        selection = ((df.snc.values == 1) & (df.sst.values >= -1.8))

        count += 1
        row = []

        grid, bias_cov, deck_cov = grid_selection(df, selection, climatology, sampling_unc, separates=True)
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

        biases = interp.project_covariance(bias_cov)
        biases.data5[np.isnan(sampling_unc.sst.values[0:1, :, :])] = np.nan
        biases.plot_map_5x5(filename=data_dir / "ICOADS" / "Figures" / f"biases_{year}{month:02d}.png")

        deck_biases = interp.project_covariance(deck_cov)
        deck_biases.data5[np.isnan(sampling_unc.sst.values[0:1, :, :])] = np.nan
        deck_biases.plot_map_5x5(filename=data_dir / "ICOADS" / "Figures" / f"deck_biases_{year}{month:02d}.png")

        all_data[count, :, :] = grid.data5[0, :, :]
        all_interpolate[count, :, :] = interpolated_grid.data5[0, :, :]
        all_nobs[count, :, :] = grid.numobs5[0, :, :]
        all_unc[count, :, :] = grid.unc5[0, :, :]

        # Plot some progress plots
        grid.plot_map_1x1(filename=data_dir / "ICOADS" / "Figures" / f"one_deg_{year}{month:02d}.png")
        grid.plot_map_5x5(filename=data_dir / "ICOADS" / "Figures" / f"five_deg_{year}{month:02d}.png")
        interpolated_grid.plot_map_5x5(
            filename=data_dir / "ICOADS" / "Figures" / f"five_deg_interp_{year}{month:02d}.png")
        grid.plot_map_unc_5x5(filename=data_dir / "ICOADS" / "Figures" / f"unc_{year}{month:02d}.png")
        interpolated_grid.plot_map_unc_5x5(
            filename=data_dir / "ICOADS" / "Figures" / f"unc_interp_{year}{month:02d}.png")

        # Calculate the area average for the grid
        ts.append(gmsst)
        ts_unc.append(gmsst_unc)
        time.append(year + (month - 1) / 12.)

        # Just ships
        selection = (df.snc.values == 1) & (df.pt.values != 6) & (df.pt.values != 7)
        grid = grid_selection(df, selection, climatology, sampling_unc, constant=0.2)
        for key, entry in regions.items():
            gmsst, gmsst_unc = grid.calculate_area_average_with_covariance(
                areas=areas, lat_range=entry["lat_range"], lon_range=entry["lon_range"]
            )
            row.append(gmsst)
            row.append(gmsst_unc)
        ship_data[count, :, :] = grid.data5[0, :, :]
        ship_nobs[count, :, :] = grid.numobs5[0, :, :]
        ship_unc[count, :, :] = grid.unc5[0, :, :]

        ship_grid = grid

        # Just drifters
        selection = (df.snc.values == 1) & (df.pt.values == 7)
        grid = grid_selection(df, selection, climatology, sampling_unc)
        for key, entry in regions.items():
            gmsst, gmsst_unc = grid.calculate_area_average_with_covariance(
                areas=areas, lat_range=entry["lat_range"], lon_range=entry["lon_range"]
            )
            row.append(gmsst)
            row.append(gmsst_unc)
        drifter_data[count, :, :] = grid.data5[0, :, :]
        drifter_nobs[count, :, :] = grid.numobs5[0, :, :]
        drifter_unc[count, :, :] = grid.unc5[0, :, :]

        for key, entry in regions.items():
            gmsst, gmsst_unc = interpolated_grid.calculate_area_average_with_covariance(
                areas=areas, lat_range=entry["lat_range"], lon_range=entry["lon_range"]
            )
            row.append(gmsst)
            row.append(gmsst_unc)
        interp_data[count, :, :] = interpolated_grid.data5[0, :, :]
        interp_unc[count, :, :] = interpolated_grid.unc5[0, :, :]

        time_series.loc[count] = row
        time_series.to_csv(data_dir / "ICOADS" / "OutputData" / "timeseries_with_uncertainty.csv")

    avships = time_series['ship']
    avships_unc = time_series['ship_unc']
    plt.fill_between(
        time, avships['Global'] + 2 * avships_unc['Global'], avships['Global'] - 2 * avships_unc['Global'],
        label="Ships", color="blue", alpha=0.5
    )

    avdrifters = time_series['drifter']
    avdrifters_unc = time_series['drifter_unc']
    plt.fill_between(
        time, avdrifters['Global'] + 2 * avdrifters_unc['Global'],
              avdrifters['Global'] - 2 * avdrifters_unc['Global'], label="Drifters", color="orange", alpha=0.5
    )

    avinterp = time_series['interp']
    avinterp_unc = time_series['interp_unc']
    plt.fill_between(
        time, avinterp['Global'] + 2 * avinterp_unc['Global'], avinterp['Global'] - 2 * avinterp_unc['Global'],
        label="Interpolated", color="red", alpha=0.5
    )

    plt.xlim(1850, 2027)
    plt.ylim(-1.0, 0.85)

    plt.legend()
    plt.savefig(data_dir / "ICOADS" / "Figures" / "timeseries_with_uncertainty.png")

    time_series.to_csv(data_dir / "ICOADS" / "OutputData" / "timeseries_with_uncertainty.csv")

    # Transfer the data to xarray DataArrays and write out
    all_data = all_data[0:count + 1, :, :]
    all_interpolate = all_interpolate[0:count + 1, :, :]
    all_unc = all_unc[0:count + 1, :, :]
    all_nobs = all_nobs[0:count + 1, :, :]

    ship_data = ship_data[0:count + 1, :, :]
    ship_unc = ship_unc[0:count + 1, :, :]
    ship_nobs = ship_nobs[0:count + 1, :, :]

    drifter_data = drifter_data[0:count + 1, :, :]
    drifter_unc = drifter_unc[0:count + 1, :, :]
    drifter_nobs = drifter_nobs[0:count + 1, :, :]

    interp_data = interp_data[0:count + 1, :, :]
    interp_unc = interp_unc[0:count + 1, :, :]

    date_range = pd.date_range(start=f'1850-01-01', freq='1MS', periods=count + 1)

    oo_anomalies = gridder.Grid.make_xarray(all_data, res=5, times=date_range)
    oo_interpolated = gridder.Grid.make_xarray(all_interpolate, res=5, times=date_range)
    oo_uncertainty = gridder.Grid.make_xarray(all_unc, res=5, times=date_range)
    oo_numobs = gridder.Grid.make_xarray(all_nobs, res=5, times=date_range)

    oo_anomalies.to_netcdf(data_dir / "ICOADS" / "oo_anomalies.nc")
    oo_interpolated.to_netcdf(data_dir / "ICOADS" / "oo_interpolated.nc")
    oo_uncertainty.to_netcdf(data_dir / "ICOADS" / "oo_uncertainty.nc")
    oo_numobs.to_netcdf(data_dir / "ICOADS" / "oo_numobs.nc")

    oo_anomalies = gridder.Grid.make_xarray(ship_data, res=5, times=date_range)
    oo_uncertainty = gridder.Grid.make_xarray(ship_unc, res=5, times=date_range)
    oo_numobs = gridder.Grid.make_xarray(ship_nobs, res=5, times=date_range)

    oo_anomalies.to_netcdf(data_dir / "ICOADS" / "oo_anomalies_ship.nc")
    oo_uncertainty.to_netcdf(data_dir / "ICOADS" / "oo_uncertainty_ship.nc")
    oo_numobs.to_netcdf(data_dir / "ICOADS" / "oo_numobs_ship.nc")

    oo_anomalies = gridder.Grid.make_xarray(drifter_data, res=5, times=date_range)
    oo_uncertainty = gridder.Grid.make_xarray(drifter_unc, res=5, times=date_range)
    oo_numobs = gridder.Grid.make_xarray(drifter_nobs, res=5, times=date_range)

    oo_anomalies.to_netcdf(data_dir / "ICOADS" / "oo_anomalies_drifter.nc")
    oo_uncertainty.to_netcdf(data_dir / "ICOADS" / "oo_uncertainty_drifter.nc")
    oo_numobs.to_netcdf(data_dir / "ICOADS" / "oo_numobs_drifter.nc")

    oo_anomalies = gridder.Grid.make_xarray(interp_data, res=5, times=date_range)
    oo_uncertainty = gridder.Grid.make_xarray(interp_unc, res=5, times=date_range)

    oo_anomalies.to_netcdf(data_dir / "ICOADS" / "oo_anomalies_interp_adjusted.nc")
    oo_uncertainty.to_netcdf(data_dir / "ICOADS" / "oo_uncertainty_interp_adjusted.nc")
