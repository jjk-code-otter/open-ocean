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
import copy
from pathlib import Path
import os
import shutil
import requests
from datetime import datetime
import xarray as xr
import pandas as pd
import numpy as np
import open_ocean.gridder as gridder

DATA_DIR = Path(os.getenv("OODIR"))


def get_file(year, month, day):
    url = (
        f"https://dap.ceda.ac.uk/neodc/eocis/data/global_and_regional/sea_surface_temperature/"
        f"CDR_v3/Analysis/L4/v3.0.1/"
        f"{year}/{month:02d}/{day:02d}/"
        f"{year}{month:02d}{day:02d}120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR3.0-v02.0-fv01.0.nc"
    )

    out_path = DATA_DIR / "SST_CCI" / f"{year}{month:02d}{day:02d}120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR3.0-v02.0-fv01.0.nc"

    if out_path.exists():
        return out_path

    r = requests.get(url, stream=True, headers={'User-agent': 'Mozilla/5.0'})

    if r.status_code == 200:
        with open(out_path, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)

    return out_path


def read_samples_from_file(year, month, day, out_path, rng, sample_denominator=1000):
    ds = xr.open_dataset(out_path)
    ds = ds.analysed_sst

    sst = ds.values.flatten()

    latitude = (np.repeat(np.reshape(ds.lat.values, (1, 3600, 1)), 7200, axis=2)).flatten()
    longitude = (np.repeat(np.reshape(ds.lon.values, (1, 1, 7200)), 3600, axis=1)).flatten()

    del ds

    latitude = latitude[~np.isnan(sst)]
    longitude = longitude[~np.isnan(sst)]
    sst = sst[~np.isnan(sst)]

    selection = rng.choice(np.arange(len(sst)), int(len(sst) / sample_denominator), replace=False)

    latitude = latitude[selection]
    longitude = longitude[selection]
    sst = sst[selection]
    d = datetime(year, month, day)
    date = [d for _ in range(len(sst))]

    return latitude, longitude, sst, date


def estimate_sampling_uncertainty(
        year,
        month,
        rng,
        all_lats,
        all_lons,
        all_dates,
        all_ssts,
        climatology,
        n_iterations=100,
        n_obs_max=10

):
    platform_id = np.array([0 for _ in range(len(all_ssts))])

    grid = gridder.Grid(
        year, month, platform_id, all_lats, all_lons, all_dates, all_ssts, platform_id, climatology
    )

    s2 = np.zeros((n_obs_max, 36, 72))
    s1 = np.zeros((n_obs_max, 36, 72))
    sigma = np.zeros((n_obs_max, 36, 72))

    # Calculate the full grid average from all observations. Can take a while
    grid.do_two_step_5x5_gridding()
    full_average = copy.deepcopy(grid.data5)

    for n_samples in range(1, n_obs_max+1):
        for i in range(n_iterations):
            print(i, n_samples)

            grid.do_one_step_5x5_sampler_gridding(n_samples=n_samples, rng=rng)

            s2[n_samples - 1, :, :] = s2[n_samples - 1, :, :] + (grid.data5[0, :, :] - full_average[0, :, :]) ** 2
            s1[n_samples - 1, :, :] = s1[n_samples - 1, :, :] + (grid.data5[0, :, :] - full_average[0, :, :])

        sigma[n_samples - 1, :, :] = np.sqrt(
            n_iterations * s2[n_samples - 1, :, :] - s1[n_samples - 1, :, :] ** 2
        ) / n_iterations

        if n_samples == 1:
            grid.unc5[0, :, :] = sigma[n_samples - 1, :, :]
            grid.plot_map_unc_5x5(levels=np.arange(0, 2.5, 0.1))

    return sigma


def process_month(year, month, rng, climatology):
    all_ssts = np.zeros((0))
    all_lats = np.zeros((0))
    all_lons = np.zeros((0))
    all_dates = []

    month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    for day in range(1, month_lengths[month - 1] + 1):
        print(day)

        out_path = get_file(year, month, day)
        latitude, longitude, sst, date = read_samples_from_file(
            year,month, day, out_path, rng, sample_denominator=1000
        )

        all_ssts = np.concatenate((all_ssts, sst))
        all_lats = np.concatenate((all_lats, latitude))
        all_lons = np.concatenate((all_lons, longitude))
        all_dates = all_dates + date

    sigma = estimate_sampling_uncertainty(
        year,
        month,
        rng,
        all_lats,
        all_lons,
        all_dates,
        all_ssts,
        climatology,
        n_obs_max=1
    )

    return sigma

if __name__ == '__main__':
    year = 2005
    climatology = xr.open_dataset(DATA_DIR / "SST_CCI_climatology" / "SST_1x1_daily.nc")
    rng = np.random.default_rng(seed=26237)

    all_sigma = np.zeros((12, 36, 72))

    for month in range(1, 13):
        sigma = process_month(year, month, rng, climatology)
        all_sigma[month-1, :, :] = sigma[0, :, :]

    grid = gridder.Grid.make_xarray(all_sigma, res=5, times=pd.date_range(start=f'1991-01-01', freq='1MS', periods=12))
    grid.to_netcdf(DATA_DIR / 'IQUAM' / 'OutputData' / 'sampling_uncertainty.nc')