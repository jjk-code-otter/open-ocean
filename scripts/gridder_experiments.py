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


def grid_selection(iquam, selection):

    id = iquam.platform_id.values[selection]
    type = iquam.platform_type.values[selection]
    lats = iquam.lat.values[selection]
    lons = iquam.lon.values[selection]
    values = iquam.sst.values[selection]

    # Convert dates
    months = iquam.month.values[selection].astype(int)
    days = iquam.day.values[selection].astype(int)
    dates = [datetime(2020, months[i], days[i]) for i in range(len(months))]

    # Grid up the data
    grid2 = gridder.Grid(2020, 10, id, lats, lons, dates, values, type, climatology)
    grid2.do_two_step_5x5_gridding()
    grid2.calculate_covariance()

    # Grid up the data
    grid = gridder.Grid(2020, 10, id, lats, lons, dates, values, type, climatology)
    grid.do_one_step_5x5_gridding()
    grid.calculate_covariance()

    return grid2, grid


if __name__ == "__main__":
    data_dir = Path(os.getenv("DATADIR"))
    coder = xr.coders.CFDatetimeCoder(time_unit="s")

    ts1 = []
    ts2 = []
    time = []
    ship_weight1 = []
    ship_weight2 = []
    drifter_weight1 = []
    drifter_weight2 = []
    unc1 = []
    unc2 = []

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

        grid_two_step, grid_one_step = grid_selection(iquam, selection)
        all_data[count, :, :] = grid_two_step.data5[0, :, :]
        all_nobs[count, :, :] = grid_two_step.numobs5[0, :, :]
        all_unc[count, :, :] = grid_two_step.unc5[0, :, :]

        ship_weight1.append(np.sum(grid_one_step.weights5[grid_one_step.type == 1]))
        ship_weight2.append(np.sum(grid_two_step.weights5[grid_two_step.type == 1]))

        drifter_weight1.append(np.sum(grid_one_step.weights5[grid_one_step.type == 2]))
        drifter_weight2.append(np.sum(grid_two_step.weights5[grid_two_step.type == 2]))

        unc1.append(np.mean(grid_one_step.unc5[~np.isnan(grid_one_step.unc5)]))
        unc2.append(np.mean(grid_two_step.unc5[~np.isnan(grid_two_step.unc5)]))

        print(f"1 step unc {unc1[-1]:.3f}")
        print(f"2 step unc {unc2[-1]:.3f}")

        print(f"1 step ship: {ship_weight1[-1]:.3f} drifter: {drifter_weight1[-1]:.3f}")
        print(f"2 step ship: {ship_weight2[-1]:.3f} drifter: {drifter_weight2[-1]:.3f}")

        # Plot some progress plots
        grid_two_step.plot_map_5x5(filename=data_dir / "IQUAM" / "Figures" / f"two_step_five_deg_{year}{month:02d}.png")
        grid_two_step.plot_map_unc_5x5(filename=data_dir / "IQUAM" / "Figures" / f"two_step_unc_{year}{month:02d}.png")

        # Calculate the area average for the grid
        gmsst1 = grid_one_step.calculate_area_average([-90, 90], [-180, 180])
        gmsst2 = grid_two_step.calculate_area_average([-90, 90], [-180, 180])
        ts1.append(gmsst1)
        ts2.append(gmsst2)
        time.append(year + (month - 1) / 12.)

        print(f"{year} {month:02d}: {gmsst1:.3f} {gmsst2:.3f} {gmsst1-gmsst2:.3f}")

    # Transfer the data to xarray DataArrays and write out
    all_data = all_data[0:count + 1, :, :]
    all_unc = all_unc[0:count + 1, :, :]
    all_nobs = all_nobs[0:count + 1, :, :]

    date_range = pd.date_range(start=f'1981-09-01', freq='1MS', periods=count + 1)

    oo_anomalies = gridder.Grid.make_xarray(all_data, res=5, times=date_range)
    oo_uncertainty = gridder.Grid.make_xarray(all_unc, res=5, times=date_range)
    oo_numobs = gridder.Grid.make_xarray(all_nobs, res=5, times=date_range)

    oo_anomalies.to_netcdf(data_dir / "IQUAM" / "oo_anomalies_twostep.nc")
    oo_uncertainty.to_netcdf(data_dir / "IQUAM" / "oo_uncertainty_twostep.nc")
    oo_numobs.to_netcdf(data_dir / "IQUAM" / "oo_numobs_twostep.nc")

    plt.plot(time, ship_weight1, label="Ship weight one-step")
    plt.plot(time, ship_weight2, label="Ship weight two-step")
    plt.plot(time, drifter_weight1, label="Drifter weight one-step")
    plt.plot(time, drifter_weight2, label="Drifter weight two-step")
    plt.legend()
    plt.xlim(1979, 2012)
    plt.ylim(0.0, 1000)
    plt.savefig(data_dir / "IQUAM" / "Figures" / "weights_comparison_two_step.png")
    plt.close()

    plt.plot(time, unc1, label="One step")
    plt.plot(time, unc2, label="Two step")
    plt.legend()
    plt.xlim(1979, 2012)
    plt.ylim(0.0, 0.5)
    plt.savefig(data_dir / "IQUAM" / "Figures" / "unc_comparison_two_step.png")
    plt.close()

    # Summary plot
    plt.plot(time, ts1, label="1-step grid")
    plt.plot(time, ts2, label="2-step grid")
    plt.legend()
    plt.xlim(1979, 2012)
    plt.savefig(data_dir / "IQUAM" / "Figures" / "IQUAM_grid_average_time_series_two_step.png")
    plt.close()
