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
import copy
import pandas as pd
import numpy as np
from pathlib import Path
import calendar
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from open_ocean.utils import convert_climatology_to_ocean_areas, convert_dates


def plot_4_up(grid1, grid2, grid3, grid4, titles, filename):
    gridx = [
        gridder.Grid.make_xarray(grid1.data5, res=5),
        gridder.Grid.make_xarray(grid2.data5, res=5),
        gridder.Grid.make_xarray(grid3.data5, res=5),
        gridder.Grid.make_xarray(grid4.data5, res=5)
    ]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 9), subplot_kw=dict(projection=ccrs.PlateCarree()))
    plt.subplots_adjust(wspace=0, hspace=0)

    xx = 0
    yy = 0
    for i, g in enumerate(gridx):
        longitude = g.sst.longitude
        latitude = g.sst.latitude
        axes[xx, yy].coastlines(lw=1, color='black')
        axes[xx, yy].pcolormesh(longitude, latitude, g.sst[0], vmin=-3.0, vmax=3.0, cmap='RdBu_r')
        axes[xx, yy].text(-175, 77, titles[i], color='black')

        xx += 1
        if xx > 1:
            xx = 0
            yy += 1

    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def plot_more_up(grid_list, titles, types, filename):
    gridx = []
    for i, g in enumerate(grid_list):
        if types[i] == 'anom':
            gridx.append(gridder.Grid.make_xarray(g.data5, res=5))
        elif types[i] == 'unc':
            gridx.append(gridder.Grid.make_xarray(g.unc5, res=5))
        elif types[i] == 'numobs':
            gridx.append(gridder.Grid.make_xarray(g.numobs5, res=5))

    ngrids = len(grid_list)

    nrows = 2
    ncols = 2

    if ngrids == 5 or ngrids == 6:
        nrows = 2
        ncols = 3
    if ngrids > 6 and ngrids <= 9:
        nrows = 3
        ncols = 3
    if ngrids > 9:
        nrows = 4
        ncols = 4

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 9), subplot_kw=dict(projection=ccrs.PlateCarree()))
    plt.subplots_adjust(wspace=0, hspace=0)

    xx = 0
    yy = 0
    for i, g in enumerate(gridx):
        longitude = g.sst.longitude
        latitude = g.sst.latitude
        axes[yy, xx].coastlines(lw=1, color='black')
        if types[i] == 'anom':
            axes[yy, xx].pcolormesh(
                longitude, latitude, g.sst[0], vmin=-3.0, vmax=3.0, cmap='RdBu_r'
            )
        elif types[i] == 'unc':
            axes[yy, xx].pcolormesh(
                longitude, latitude, g.sst[0], vmin=0, vmax=1.0, cmap='viridis'
            )
        elif types[i] == 'numobs':
            axes[yy, xx].pcolormesh(
                longitude, latitude, g.sst[0], vmin=0, vmax=10, cmap='viridis'
            )

        axes[yy, xx].text(-175, 77, titles[i], color='black')

        xx += 1
        if xx >= ncols:
            xx = 0
            yy += 1

    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def fix_ice_array(month, ice_ym, threshold_ice_fraction=0.9):
    """Coerce the ice array into the necessary shape and extrac the values as SSTs"""
    ice_ym = ice_ym.sic.values
    ice_ym = np.flip(ice_ym, 0)  # latitudes are specified upside down
    ice_ym = np.roll(ice_ym, 180, 1)  # longitudes start at 0 not -180

    latitude = np.linspace(-89.5, 89.5, 180)
    latitude = np.reshape(latitude, (180, 1))
    latitude = np.repeat(latitude, 360, 1)

    longitude = np.linspace(-179.5, 179.5, 360)
    longitude = np.reshape(longitude, (1, 360))
    longitude = np.repeat(longitude, 180, 0)

    selection = (ice_ym >= threshold_ice_fraction)

    ice_ym = ice_ym[selection]
    longitude = longitude[selection]
    latitude = latitude[selection]

    months = np.array([month for _ in range(len(ice_ym))])
    days = np.array([14 for _ in range(len(ice_ym))])
    dates = convert_dates(months, days)

    id = ['ICE' for _ in range(len(ice_ym))]

    values = [273.15 - 1.8 for _ in range(len(ice_ym))]
    type = [99 for _ in range(len(ice_ym))]
    deck = [-1 for _ in range(len(ice_ym))]

    return longitude, latitude, dates, id, values, type, deck


class IcoadsGridder:

    def __init__(
            self,
            year, month, df, climatology, sampling_unc, tracking=True
    ):
        self.year = year
        self.month = month
        self.df = df
        self.climatology = climatology
        self.sampling_unc = sampling_unc
        self.tracking = tracking
        self.processed = {}

    def make_selection(self, selection):
        # Exclude observations from decks 874 (they're a mess) and 780 (subsurface data)
        deck = self.df.dck.values
        selection = selection & (deck != 874)
        selection = selection & (deck != 780)

        type = self.df.pt.values[selection]
        lats = self.df.lat.values[selection]
        lons = self.df.lon.values[selection]
        values = self.df.sst.values[selection] + 273.15
        days = self.df.day.values[selection]
        months = self.df.month.values[selection]
        deck = self.df.dck.values[selection]

        # If we are using the Kent tracking IDs then we need to copy in the drifter and mooring IDs from ICOADS
        if self.tracking:
            pid = self.df.trackid.values[selection]
            icoads_id = self.df.id.values[selection]
            pid[type == 7] = icoads_id[type == 7]
            pid[type == 6] = icoads_id[type == 6]

        else:
            pid = self.df.id.values[selection]

        # Drifters and moorings don't have deck biases
        deck[type == 7] = -2
        deck[type == 6] = -2

        # ICOADS longitudes are specified in the range -180 to 360 but we want -180 to 180.
        lons[lons > 180.0] = lons[lons > 180.0] - 360.0

        # ICOAD has different platform type identifiers to IQUAM types that the code expects.
        pt_copy = copy.deepcopy(type)
        # IQUAM platform types
        SHIP = 1
        DRIFT = 2
        MOOR = 3
        ARGO = 5

        pt_copy[:] = SHIP
        pt_copy[type == 7] = DRIFT
        pt_copy[type == 6] = MOOR

        type = pt_copy

        # Some ICOADS observations have bad dates
        month_lengths = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        if calendar.isleap(df.year.values[0]):
            month_lengths[1] = 29
        valid_days = (days > 0) & (days <= month_lengths[months - 1])

        pid = pid[valid_days]
        type = type[valid_days]
        lats = lats[valid_days]
        lons = lons[valid_days]
        values = values[valid_days]
        months = months[valid_days]
        days = days[valid_days]
        deck = deck[valid_days]

        # Japanese truncated data (Chan et al. 2019)
        values[deck == 118] = values[deck == 118] + 0.5
        values[deck == 119] = values[deck == 119] + 0.5

        # Convert dates
        dates = convert_dates(months.astype(int), days.astype(int))

        self.processed = {
            "id": pid,
            "type": type,
            "lats": lats,
            "lons": lons,
            "dates": dates,
            "months": months,
            "days": days,
            "values": values,
            "deck": deck,
        }

    def grid_selection(self, constant=0.0, separates=False, calc_deck_level_cov=False):
        # Grid up the data
        self.grid = gridder.Grid(
            2020,
            10,
            self.processed['id'],
            self.processed['lats'],
            self.processed['lons'],
            self.processed['dates'],
            self.processed['values'],
            self.processed['type'],
            self.climatology
        )

        self.grid.add_sampling_uncertainties(self.sampling_unc)
        self.grid.do_1x1_gridding()
        self.grid.do_one_step_5x5_gridding()

        self.bias_cov = grid.calculate_covariance(constant=constant, separates=separates)

        if calc_deck_level_cov:
            self.deck_cov = self.grid.add_correlated_error(
                'deck',
                self.processed['deck'],
                0.2,
                exclusions=[
                    -1,  # ice
                    -2,  # buoys
                ]
            )
        else:
            self.deck_cov = np.zeros((2592, 2592))


def grid_selection(
        year,
        month,
        df,
        selection,
        climatology,
        sampling_unc,
        constant=None,
        separates=False,
        calc_deck_level_cov=False,
        tracking=True,
        ice=None,
) -> gridder.Grid:
    """
    Grid a particular selection of data

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame containing the data to be gridded. Required columns 'pt', 'lat', 'lon', 'dck', 'day', 'month', 'sst',
        'id'. If tracking is set, then 'trackid' is also required
    selection: np.ndarray
        Array containing the selection of data to be gridded
    climatology: xarray.DataArray
        SST climatology
    sampling_unc: np.ndarray
        Array containing the sampling uncertainty for one observations. Shape (36,72)
    constant: float or None
        Constant value to be added to the covariance matrix.
    separates: bool
        If set to True, return the bias and deck covariances in addition to the grid
    tracking: bool
        If set to True, covariances are calculated using the `trackid` instead of the ICOADS `id`.
    ice: xarray.DataArray or None
        Ice data set or None

    Returns
    -------
    Grid or (Grid, np.ndarray, np.ndarray)
        Return the gridded data or the gridded data and two covariance matrices.
    """
    # Exclude observations from decks 874 (they're a mess) and 780 (subsurface data)
    deck = df.dck.values
    selection = selection & (deck != 874)
    selection = selection & (deck != 780)

    type = df.pt.values[selection]
    lats = df.lat.values[selection]
    lons = df.lon.values[selection]
    values = df.sst.values[selection] + 273.15
    days = df.day.values[selection]
    months = df.month.values[selection]
    deck = df.dck.values[selection]

    # If we are using the Kent tracking IDs then we need to copy in the drifter and mooring IDs from ICOADS
    if tracking:
        pid = df.trackid.values[selection]
        icoads_id = df.id.values[selection]
        pid[type == 7] = icoads_id[type == 7]
        pid[type == 6] = icoads_id[type == 6]

    else:
        pid = df.id.values[selection]

    # Drifters and moorings don't have deck biases
    deck[type == 7] = -2
    deck[type == 6] = -2

    # ICOADS longitudes are specified in the range -180 to 360 but we want -180 to 180.
    lons[lons > 180.0] = lons[lons > 180.0] - 360.0

    # ICOAD has different platform type identifiers to IQUAM types that the code expects.
    pt_copy = copy.deepcopy(type)
    # IQUAM platform types
    SHIP = 1
    DRIFT = 2
    MOOR = 3
    ARGO = 5

    pt_copy[:] = SHIP
    pt_copy[type == 7] = DRIFT
    pt_copy[type == 6] = MOOR

    type = pt_copy

    # Some ICOADS observations have bad dates
    month_lengths = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    if calendar.isleap(df.year.values[0]):
        month_lengths[1] = 29
    valid_days = (days > 0) & (days <= month_lengths[months - 1])

    pid = pid[valid_days]
    type = type[valid_days]
    lats = lats[valid_days]
    lons = lons[valid_days]
    values = values[valid_days]
    months = months[valid_days]
    days = days[valid_days]
    deck = deck[valid_days]

    # Japanese truncated data (Chan et al. 2019)
    values[deck == 118] = values[deck == 118] + 0.5
    values[deck == 119] = values[deck == 119] + 0.5

    # Convert dates
    dates = convert_dates(months.astype(int), days.astype(int))

    # Add ice by adding "pseudo obs" at the grid cell centres of cells containing ice of greater than
    # threshold_ice_fraction. Anything with a sea ice concentration above that is set to -1.8C.
    if ice is not None:
        ice_lon, ice_lat, ice_dates, ice_id, ice_values, ice_type, ice_deck = fix_ice_array(
            month, ice, threshold_ice_fraction=0.9
        )
        pid = np.concatenate([pid, ice_id])
        type = np.concatenate([type, ice_type])
        lats = np.concatenate([lats, ice_lat])
        lons = np.concatenate([lons, ice_lon])
        values = np.concatenate([values, ice_values])
        dates = dates + ice_dates
        deck = np.concatenate([deck, ice_deck])

    # Grid up the data
    grid = gridder.Grid(2020, 10, pid, lats, lons, dates, values, type, climatology)
    grid.add_sampling_uncertainties(sampling_unc)
    grid.do_1x1_gridding()
    grid.do_one_step_5x5_gridding()
    bias_cov = grid.calculate_covariance(constant=constant, separates=separates)

    if calc_deck_level_cov:
        deck_cov = grid.add_correlated_error(
            'deck',
            deck,
            0.2,
            exclusions=[
                -1,  # ice
                -2,  # buoys
            ]
        )
    else:
        deck_cov = np.zeros((2592, 2592))

    if separates:
        return grid, bias_cov, deck_cov

    return grid


if __name__ == '__main__':
    data_dir = Path(os.getenv("OODIR"))  #

    start_year = 1850
    end_year = 2005

    ts = []
    ts_unc = []
    time = []

    with open('regions.json', 'r') as f:
        regions = json.load(f)

    climatology = xr.open_dataset(data_dir / "SST_CCI_climatology" / "SST_1x1_daily.nc")
    areas = convert_climatology_to_ocean_areas(climatology)
    sampling_unc = xr.open_dataset(data_dir / "IQUAM" / "OutputData" / "sampling_uncertainty.nc")
    ice = xr.open_dataset(data_dir / "IQUAM" / "InputData" / "HadISST.2.2.0.0_sea_ice_concentration.nc",
                          engine='netcdf4')

    n_time = (end_year - start_year + 1) * 12

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

    interp2_data = np.zeros((n_time, 36, 72)) + np.nan
    interp2_unc = np.zeros((n_time, 36, 72)) + np.nan

    region_names = [key for key in regions.keys()]
    component_names = [
        "all", "all_unc",
        "ship", "ship_unc",
        "drifter", "drifter_unc",
        "interp", "interp_unc",
        "interp2", "interp2_unc"
    ]

    mux = pd.MultiIndex.from_product([component_names, region_names])
    time_series = pd.DataFrame(columns=mux)

    count = -1

    for year, month in product(range(start_year, end_year + 1), range(1, 13)):
        print(year, month)
        iceym = ice.sel(time=f"{year}-{month:02d}-15", method="nearest")

        file = data_dir / "ICOADS" / f"icoads_{year}{month:02d}.csv"

        df = pd.read_csv(file)

        selection = ((df.snc.values == 1) & (df.sst.values >= -1.8))

        count += 1
        row = []

        grid, bias_cov, deck_cov = grid_selection(
            year,
            month,
            df,
            selection,
            climatology,
            sampling_unc,
            separates=True,
            calc_deck_level_cov=True
        )
        grid.tidy_grid()
        for key, entry in regions.items():
            gmsst, gmsst_unc = grid.calculate_area_average_with_covariance(
                areas=areas, lat_range=entry["lat_range"], lon_range=entry["lon_range"]
            )
            row.append(gmsst)
            row.append(gmsst_unc)
            print(f"{key} {year} {month:02d}: {gmsst:.3f} ± {gmsst_unc:.3f}")

        basic_grid = grid
        print(np.max(basic_grid.data5[~np.isnan(basic_grid.data5)]), np.min(basic_grid.data5[~np.isnan(basic_grid.data5)]))

        kernel = io.Kernel(0.6, 1300.0, 1.5)
        interp = io.GPInterpolator(grid, kernel)
        interp.make_covariance(constant=0.5)

        accumulated_spherical_cov = np.zeros((2592, 2592)) + 0.5 * 0.5
        for n in range(1, 4):
            for m in range(-1 * n, n):
                sph_cov = interp.add_spherical_harmonics_to_covariance(n, m, 0.2)
                accumulated_spherical_cov += sph_cov

        interpolated_grid = interp.do_interpolation()
        interpolated_grid.data5[np.isnan(sampling_unc.sst.values[0:1, :, :])] = np.nan

        spheroidal = interp.project_covariance(accumulated_spherical_cov)
        spheroidal.data5[np.isnan(sampling_unc.sst.values[0:1, :, :])] = np.nan

        biases = interp.project_covariance(bias_cov)
        biases.data5[np.isnan(sampling_unc.sst.values[0:1, :, :])] = np.nan

        deck_biases = interp.project_covariance(deck_cov)
        deck_biases.data5[np.isnan(sampling_unc.sst.values[0:1, :, :])] = np.nan

        all_data[count, :, :] = grid.data5[0, :, :]
        all_interpolate[count, :, :] = interpolated_grid.data5[0, :, :]
        all_nobs[count, :, :] = grid.numobs5[0, :, :]
        all_unc[count, :, :] = grid.unc5[0, :, :]

        # Calculate the area average for the grid
        ts.append(gmsst)
        ts_unc.append(gmsst_unc)
        time.append(year + (month - 1) / 12.)

        # Just ships
        selection = (df.snc.values == 1) & (df.pt.values != 6) & (df.pt.values != 7)
        grid = grid_selection(
            year,
            month,
            df,
            selection,
            climatology,
            sampling_unc,
            constant=0.2,
            calc_deck_level_cov=True
        )
        grid.tidy_grid()
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

        # Just drifters and moorings
        selection = (df.snc.values == 1) & ((df.pt.values == 7) | (df.pt.values == 6))
        grid = grid_selection(
            year,
            month,
            df,
            selection,
            climatology,
            sampling_unc,
            calc_deck_level_cov=False
        )
        grid.tidy_grid()
        for key, entry in regions.items():
            gmsst, gmsst_unc = grid.calculate_area_average_with_covariance(
                areas=areas, lat_range=entry["lat_range"], lon_range=entry["lon_range"]
            )
            row.append(gmsst)
            row.append(gmsst_unc)
        drifter_data[count, :, :] = grid.data5[0, :, :]
        drifter_nobs[count, :, :] = grid.numobs5[0, :, :]
        drifter_unc[count, :, :] = grid.unc5[0, :, :]

        drifter_grid = grid
        drifter_cell_count = np.count_nonzero(drifter_grid.numobs5)

        for key, entry in regions.items():
            gmsst, gmsst_unc = interpolated_grid.calculate_area_average_with_covariance(
                areas=areas, lat_range=entry["lat_range"], lon_range=entry["lon_range"]
            )
            row.append(gmsst)
            row.append(gmsst_unc)
        interp_data[count, :, :] = interpolated_grid.data5[0, :, :]
        interp_unc[count, :, :] = interpolated_grid.unc5[0, :, :]

        drifter_threshold = 500
        if drifter_cell_count > drifter_threshold:
            # Do some interpolation stuff here
            kernel = io.Kernel(0.6, 1300.0, 1.5)
            interp1 = io.GPInterpolator(drifter_grid, kernel)
            interp1.make_covariance(constant=0.5)
            for n in range(1, 4):
                for m in range(-1 * n, n):
                    interp1.add_spherical_harmonics_to_covariance(n, m, 0.2)

            interpolated_grid1 = interp1.do_interpolation()
            interpolated_grid1.data5[np.isnan(sampling_unc.sst.values[0:1, :, :])] = np.nan

            ship_grid = ship_grid - interpolated_grid1

        kernel = io.Kernel(0.6, 1300.0, 1.5)
        interp2 = io.GPInterpolator(ship_grid, kernel)
        if drifter_cell_count > drifter_threshold:
            interp2.replace_covariance(interp1.posterior)
        else:
            interp2.make_covariance(constant=0.5)
            for n in range(1, 4):
                for m in range(-1 * n, n):
                    interp2.add_spherical_harmonics_to_covariance(n, m, 0.2)

        interpolated_grid2 = interp2.do_interpolation()
        interpolated_grid2.data5[np.isnan(sampling_unc.sst.values[0:1, :, :])] = np.nan

        if drifter_cell_count > drifter_threshold:
            interpolated_grid2 = interpolated_grid2 + interpolated_grid1

        for key, entry in regions.items():
            gmsst, gmsst_unc = interpolated_grid2.calculate_area_average_with_covariance(
                areas=areas, lat_range=entry["lat_range"], lon_range=entry["lon_range"]
            )
            row.append(gmsst)
            row.append(gmsst_unc)
        interp2_data[count, :, :] = interpolated_grid2.data5[0, :, :]
        interp2_unc[count, :, :] = interpolated_grid2.unc5[0, :, :]

        plot_more_up(
            [
                basic_grid,
                basic_grid,
                basic_grid,
                interpolated_grid,
                biases,
                deck_biases,
                spheroidal,
                interpolated_grid2
            ],
            [
                f'{year}-{month:02d} Basic grid',
                'uncertainty',
                'numobs',
                'Interpolated grid',
                'Individual ship biases',
                'Deck biases',
                'Spherical Harmonics',
                'Interpolated grid adjusted',
            ],
            [
                'anom',
                'unc',
                'numobs',
                'anom',
                'anom',
                'anom',
                'anom',
                'anom'
            ],
            data_dir / "ICOADS" / "Figures" / f"four_up_{year}{month:02d}.png"
        )

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

    avinterp2 = time_series['interp2']
    avinterp2_unc = time_series['interp2_unc']
    plt.fill_between(
        time, avinterp2['Global'] + 2 * avinterp2_unc['Global'], avinterp2['Global'] - 2 * avinterp2_unc['Global'],
        label="Interpolated 2", color="green", alpha=0.5
    )

    plt.xlim(start_year - 1, end_year + 1)
    plt.ylim(-1.65, 0.75)
    plt.gcf().set_size_inches(42, 10)
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

    interp2_data = interp2_data[0:count + 1, :, :]
    interp2_unc = interp2_unc[0:count + 1, :, :]

    date_range = pd.date_range(start=f'1850-01-01', freq='1MS', periods=count + 1)

    oo_anomalies = gridder.Grid.make_xarray(all_data, res=5, times=date_range)
    oo_interpolated = gridder.Grid.make_xarray(all_interpolate, res=5, times=date_range)
    oo_uncertainty = gridder.Grid.make_xarray(all_unc, res=5, times=date_range)
    oo_numobs = gridder.Grid.make_xarray(all_nobs, res=5, times=date_range)

    oo_anomalies.to_netcdf(data_dir / "ICOADS" / "OutputData" / "oo_anomalies.nc")
    oo_interpolated.to_netcdf(data_dir / "ICOADS" / "OutputData" / "oo_interpolated.nc")
    oo_uncertainty.to_netcdf(data_dir / "ICOADS" / "OutputData" / "oo_uncertainty.nc")
    oo_numobs.to_netcdf(data_dir / "ICOADS" / "OutputData" / "oo_numobs.nc")

    oo_anomalies = gridder.Grid.make_xarray(ship_data, res=5, times=date_range)
    oo_uncertainty = gridder.Grid.make_xarray(ship_unc, res=5, times=date_range)
    oo_numobs = gridder.Grid.make_xarray(ship_nobs, res=5, times=date_range)

    oo_anomalies.to_netcdf(data_dir / "ICOADS" / "OutputData" / "oo_anomalies_ship.nc")
    oo_uncertainty.to_netcdf(data_dir / "ICOADS" / "OutputData" / "oo_uncertainty_ship.nc")
    oo_numobs.to_netcdf(data_dir / "ICOADS" / "OutputData" / "oo_numobs_ship.nc")

    oo_anomalies = gridder.Grid.make_xarray(drifter_data, res=5, times=date_range)
    oo_uncertainty = gridder.Grid.make_xarray(drifter_unc, res=5, times=date_range)
    oo_numobs = gridder.Grid.make_xarray(drifter_nobs, res=5, times=date_range)

    oo_anomalies.to_netcdf(data_dir / "ICOADS" / "OutputData" / "oo_anomalies_drifter.nc")
    oo_uncertainty.to_netcdf(data_dir / "ICOADS" / "OutputData" / "oo_uncertainty_drifter.nc")
    oo_numobs.to_netcdf(data_dir / "ICOADS" / "OutputData" / "oo_numobs_drifter.nc")

    oo_anomalies = gridder.Grid.make_xarray(interp_data, res=5, times=date_range)
    oo_uncertainty = gridder.Grid.make_xarray(interp_unc, res=5, times=date_range)

    oo_anomalies.to_netcdf(data_dir / "ICOADS" / "OutputData" / "oo_anomalies_interp.nc")
    oo_uncertainty.to_netcdf(data_dir / "ICOADS" / "OutputData" / "oo_uncertainty_interp.nc")

    oo_anomalies = gridder.Grid.make_xarray(interp2_data, res=5, times=date_range)
    oo_uncertainty = gridder.Grid.make_xarray(interp2_unc, res=5, times=date_range)

    oo_anomalies.to_netcdf(data_dir / "ICOADS" / "OutputData" / "oo_anomalies_interp_adjusted.nc")
    oo_uncertainty.to_netcdf(data_dir / "ICOADS" / "OutputData" / "oo_uncertainty_interp_adjusted.nc")
