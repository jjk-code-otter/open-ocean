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
from itertools import product
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime
import matplotlib.pyplot as plt


def convert_dates(months, days):
    return [datetime(2020, months[i], days[i]) for i in range(len(months))]

def grid_selection(iquam, selection):
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

    # Grid up the data
    grid = gridder.Grid(2020, 10, id, lats, lons, dates, values, type, climatology)
    grid.do_one_step_5x5_gridding_with_covariance()

    return grid


if __name__ == "__main__":
    data_dir = Path(os.getenv("DATADIR"))
    coder = xr.coders.CFDatetimeCoder(time_unit="s")

    ts = []
    time = []

    hadsst4 = xr.open_dataset(data_dir / "ManagedData" / "Data" / "HadSST4" / "HadSST.4.1.1.0_median.nc")

    lat_slice = hadsst4.tos.sel(latitude=slice(-90, 90))
    lat_lon_slice = lat_slice.sel(longitude=slice(-180, 180))
    weights = np.cos(np.deg2rad(lat_lon_slice.latitude))
    weighted_mean = lat_lon_slice.weighted(weights).mean(("longitude", "latitude"))
    h4_time = np.arange(len(weighted_mean)) / 12. + 1850.

    climatology = xr.open_dataset(data_dir / "SST_CCI_climatology" / "SST_1x1_daily.nc")

    n_time = (2025 - 1981 + 1) * 12

    all_data = np.zeros((n_time, 36, 72)) + np.nan
    all_nobs = np.zeros((n_time, 36, 72))
    all_unc = np.zeros((n_time, 36, 72)) + np.nan

    ship_data = np.zeros((n_time, 36, 72)) + np.nan
    ship_nobs = np.zeros((n_time, 36, 72))
    ship_unc = np.zeros((n_time, 36, 72)) + np.nan

    drifter_data = np.zeros((n_time, 36, 72)) + np.nan
    drifter_nobs = np.zeros((n_time, 36, 72))
    drifter_unc = np.zeros((n_time, 36, 72)) + np.nan

    argo_data = np.zeros((n_time, 36, 72)) + np.nan
    argo_nobs = np.zeros((n_time, 36, 72))
    argo_unc = np.zeros((n_time, 36, 72)) + np.nan

    count = -1

    for year, month in product(range(1981, 2011), range(1, 13)):
        file = data_dir / 'IQUAM' / f'{year}{month:02d}-STAR-L2i_GHRSST-SST-iQuam-V2.10-v01.0-fv01.0.nc'

        if not (file.exists()):
            continue

        iquam = xr.open_dataset(file, decode_timedelta=coder)

        # Select only high quality observations
        quality = iquam.quality_level.values
        selection = quality >= 4

        count += 1

        grid = grid_selection(iquam, selection)
        all_data[count, :, :] = grid.data5[0, :, :]
        all_nobs[count, :, :] = grid.numobs5[0, :, :]
        all_unc[count, :, :] = grid.unc[0, :, :]

        # Plot some progress plots
        grid.plot_map(filename=data_dir / "IQUAM" / "Figures" / f"one_deg_{year}{month:02d}.png")
        grid.plot_map5(filename=data_dir / "IQUAM" / "Figures" / f"five_deg_{year}{month:02d}.png")
        grid.plot_map_unc5(filename=data_dir / "IQUAM" / "Figures" / f"unc_{year}{month:02d}.png")

        # Calculate the area average for the grid
        gmsst = grid.calculate_area_average([-90, 90], [-180, 180])
        ts.append(gmsst)
        time.append(year + (month - 1) / 12.)

        print(f"{year} {month:02d}: {gmsst:.3f}")

        # Just ships
        selection = (quality >= 4) & (iquam.platform_type.values == 1)
        grid = grid_selection(iquam, selection)
        ship_data[count, :, :] = grid.data5[0, :, :]
        ship_nobs[count, :, :] = grid.numobs5[0, :, :]
        ship_unc[count, :, :] = grid.unc[0, :, :]

        # Just drifters
        selection = (quality >= 4) & (iquam.platform_type.values == 2)
        grid = grid_selection(iquam, selection)
        drifter_data[count, :, :] = grid.data5[0, :, :]
        drifter_nobs[count, :, :] = grid.numobs5[0, :, :]
        drifter_unc[count, :, :] = grid.unc[0, :, :]

        # Just Argo
        selection = (quality >= 4) & (iquam.platform_type.values == 5)
        grid = grid_selection(iquam, selection)
        argo_data[count, :, :] = grid.data5[0, :, :]
        argo_nobs[count, :, :] = grid.numobs5[0, :, :]
        argo_unc[count, :, :] = grid.unc[0, :, :]

    # Transfer the data to xarray DataArrays and write out
    all_data = all_data[0:count + 1, :, :]
    all_unc = all_unc[0:count + 1, :, :]
    all_nobs = all_nobs[0:count + 1, :, :]

    ship_data = ship_data[0:count + 1, :, :]
    ship_unc = ship_unc[0:count + 1, :, :]
    ship_nobs = ship_nobs[0:count + 1, :, :]

    drifter_data = drifter_data[0:count + 1, :, :]
    drifter_unc = drifter_unc[0:count + 1, :, :]
    drifter_nobs = drifter_nobs[0:count + 1, :, :]

    argo_data = argo_data[0:count + 1, :, :]
    argo_unc = argo_unc[0:count + 1, :, :]
    argo_nobs = argo_nobs[0:count + 1, :, :]

    date_range = pd.date_range(start=f'1981-09-01', freq='1MS', periods=count + 1)

    oo_anomalies = gridder.Grid.make_xarray(all_data, res=5, times=date_range)
    oo_uncertainty = gridder.Grid.make_xarray(all_unc, res=5, times=date_range)
    oo_numobs = gridder.Grid.make_xarray(all_nobs, res=5, times=date_range)

    oo_anomalies.to_netcdf(data_dir / "IQUAM" / "oo_anomalies.nc")
    oo_uncertainty.to_netcdf(data_dir / "IQUAM" / "oo_uncertainty.nc")
    oo_numobs.to_netcdf(data_dir / "IQUAM" / "oo_numobs.nc")

    oo_anomalies = gridder.Grid.make_xarray(ship_data, res=5, times=date_range)
    oo_uncertainty = gridder.Grid.make_xarray(ship_unc, res=5, times=date_range)
    oo_numobs = gridder.Grid.make_xarray(ship_nobs, res=5, times=date_range)

    oo_anomalies.to_netcdf(data_dir / "IQUAM" / "oo_anomalies_ship.nc")
    oo_uncertainty.to_netcdf(data_dir / "IQUAM" / "oo_uncertainty_ship.nc")
    oo_numobs.to_netcdf(data_dir / "IQUAM" / "oo_numobs_ship.nc")

    oo_anomalies = gridder.Grid.make_xarray(drifter_data, res=5, times=date_range)
    oo_uncertainty = gridder.Grid.make_xarray(drifter_unc, res=5, times=date_range)
    oo_numobs = gridder.Grid.make_xarray(drifter_nobs, res=5, times=date_range)

    oo_anomalies.to_netcdf(data_dir / "IQUAM" / "oo_anomalies_drifter.nc")
    oo_uncertainty.to_netcdf(data_dir / "IQUAM" / "oo_uncertainty_drifter.nc")
    oo_numobs.to_netcdf(data_dir / "IQUAM" / "oo_numobs_drifter.nc")

    oo_anomalies = gridder.Grid.make_xarray(argo_data, res=5, times=date_range)
    oo_uncertainty = gridder.Grid.make_xarray(argo_unc, res=5, times=date_range)
    oo_numobs = gridder.Grid.make_xarray(argo_nobs, res=5, times=date_range)

    oo_anomalies.to_netcdf(data_dir / "IQUAM" / "oo_anomalies_argo.nc")
    oo_uncertainty.to_netcdf(data_dir / "IQUAM" / "oo_uncertainty_argo.nc")
    oo_numobs.to_netcdf(data_dir / "IQUAM" / "oo_numobs_argo.nc")

    # Summary plot
    plt.plot(time, ts)
    plt.plot(h4_time, weighted_mean)
    plt.xlim(1975, 2010)
    plt.savefig(data_dir / "IQUAM" / "Figures" / "IQUAM_grid_average_time_series.png")
    plt.close()
