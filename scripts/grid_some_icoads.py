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
        values[deck == 118] = values[deck == 118] + 0.45
        values[deck == 119] = values[deck == 119] + 0.45

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

        self.bias_cov = self.grid.calculate_covariance(constant=constant, separates=separates)

        if calc_deck_level_cov:
            self.deck_cov, self.deck_cov_dict = self.grid.add_correlated_error(
                'deck',
                self.processed['deck'],
                0.2,
                exclusions=[
                    -1,  # ice
                    -2,  # buoys
                ],
                full_dict=True
            )
        else:
            self.deck_cov = np.zeros((2592, 2592))
            self.deck_cov_dict = {}

        # Remove outliers
        self.grid.tidy_grid()

    def calculate_regional_averages(self, regions, areas):
        sub_row = []
        for key, entry in regions.items():
            gmsst, gmsst_unc = self.grid.calculate_area_average_with_covariance(
                areas=areas, lat_range=entry["lat_range"], lon_range=entry["lon_range"]
            )
            sub_row.append(gmsst)
            sub_row.append(gmsst_unc)
            print(f"{key}: {gmsst:.3f} ± {gmsst_unc:.3f}")

        return sub_row

    def print_stats(self):
        nonmissing = self.grid.data5[~np.isnan(self.grid.data5)]
        print(
            np.max(nonmissing),
            np.median(nonmissing),
            np.min(nonmissing)
        )


if __name__ == '__main__':
    data_dir = Path(os.getenv("OODIR"))  #

    start_year = 1850
    end_year = 2025

    drifter_threshold = 500

    ts = []
    ts_unc = []
    time = []

    with open('regions.json', 'r') as f:
        regions = json.load(f)

    climatology = xr.open_dataset(data_dir / "SST_CCI_climatology" / "SST_1x1_daily.nc")
    variability = xr.open_dataset(data_dir / "SST_CCI" / "SST_Variability_5x5.nc")
    areas = convert_climatology_to_ocean_areas(climatology)
    sampling_unc = xr.open_dataset(data_dir / "IQUAM" / "OutputData" / "sampling_uncertainty.nc")
    ice = xr.open_dataset(data_dir / "IQUAM" / "InputData" / "HadISST.2.2.0.0_sea_ice_concentration.nc",
                          engine='netcdf4')

    region_names = [key for key in regions.keys()]
    component_names = [
        "all", "all_unc",
        "ship", "ship_unc",
        "drifter", "drifter_unc",
        "interp", "interp_unc",
        "interp2", "interp2_unc",
        "interpok", "interpok_unc",
    ]

    mux = pd.MultiIndex.from_product([component_names, region_names])
    time_series = pd.DataFrame(columns=mux)

    count = -1

    for year, month in product(range(start_year, end_year + 1), range(1, 13)):
        print(year, month)
        iceym = ice.sel(time=f"{year}-{month:02d}-15", method="nearest")
        varym = variability.sst[month-1].values.flatten()

        file = data_dir / "ICOADS" / f"icoads_{year}{month:02d}.csv"

        df = pd.read_csv(file)

        selection = ((df.snc.values == 1) & (df.sst.values >= -1.8))

        count += 1
        time_series = pd.DataFrame(columns=mux)
        row = []

        basic_grid = IcoadsGridder(year, month, df, climatology, sampling_unc)
        basic_grid.make_selection(selection)
        basic_grid.grid_selection(constant=0.0, separates=True, calc_deck_level_cov=True)
        row = row + basic_grid.calculate_regional_averages(regions, areas)
        basic_grid.print_stats()

        kernel = io.Kernel(varym, 1300.0, 1.5)
        interp_ok = io.OKInterpolator(basic_grid.grid, kernel)
        interp_ok.make_covariance()
        interpolated_grid_ok = interp_ok.do_interpolation()
        interpolated_grid_ok.data5[np.isnan(sampling_unc.sst.values[0:1, :, :])] = np.nan

        kernel = io.Kernel(varym, 1300.0, 1.5)
        interp = io.OKInterpolator(basic_grid.grid, kernel)
        # build covariance
        interp.make_covariance()
        kernel_cov = interp.cov
        accumulated_spherical_cov = np.zeros((2592, 2592))
        for n in range(1, 4):
            for m in range(-1 * n, n):
                sph_cov = interp.add_spherical_harmonics_to_covariance(n, m, 0.2)
                accumulated_spherical_cov += sph_cov
        interpolated_grid = interp.do_interpolation()
        interpolated_grid.data5[np.isnan(sampling_unc.sst.values[0:1, :, :])] = np.nan

        local = interp.project_covariance(kernel_cov)
        local.data5[np.isnan(sampling_unc.sst.values[0:1, :, :])] = np.nan

        flat = interp.project_covariance(np.zeros((2592, 2592)) + 0.5 * 0.5)
        flat.data5[np.isnan(sampling_unc.sst.values[0:1, :, :])] = np.nan
        flat.data5[~np.isnan(sampling_unc.sst.values[0:1, :, :])] = interp.beta

        spheroidal = interp.project_covariance(accumulated_spherical_cov)
        spheroidal.data5[np.isnan(sampling_unc.sst.values[0:1, :, :])] = np.nan

        biases = interp.project_covariance(basic_grid.bias_cov)
        biases.data5[np.isnan(sampling_unc.sst.values[0:1, :, :])] = np.nan

        deck_biases = interp.project_covariance(basic_grid.deck_cov)
        deck_biases.data5[np.isnan(sampling_unc.sst.values[0:1, :, :])] = np.nan

        individual_decks = interp.project_covariances_from_dict(basic_grid.deck_cov_dict)
        deck_bias_dict = {}
        for key, value in individual_decks.items():
            value.data5[np.isnan(sampling_unc.sst.values[0:1, :, :])] = np.nan
            individual_decks[key] = value
            deck_bias_dict[str(key)] = [
                np.mean(value.data5[~np.isnan(value.data5)]),
                np.mean(value.unc5[~np.isnan(value.unc5)])
            ]
        deck_bias_dict['year'] = year
        deck_bias_dict['month'] = month
        json_out_file = data_dir / "ICOADS" / "OutputData" / f"decks_{year}{month:02d}.json"
        with open(json_out_file, 'w') as f:
            json.dump(deck_bias_dict, f, sort_keys=True, indent=2)

        # Just ships
        selection = (df.snc.values == 1) & (df.pt.values != 6) & (df.pt.values != 7)
        ship_grid = IcoadsGridder(year, month, df, climatology, sampling_unc)
        ship_grid.make_selection(selection)
        ship_grid.grid_selection(constant=0.2, calc_deck_level_cov=True)
        row = row + ship_grid.calculate_regional_averages(regions, areas)

        # Just drifters and moorings
        selection = (df.snc.values == 1) & ((df.pt.values == 7) | (df.pt.values == 6))
        drifter_grid = IcoadsGridder(year, month, df, climatology, sampling_unc)
        drifter_grid.make_selection(selection)
        drifter_grid.grid_selection(constant=0.0, calc_deck_level_cov=False)
        row = row + drifter_grid.calculate_regional_averages(regions, areas)

        row = row + interpolated_grid.calculate_regional_averages(regions, areas)

        # Do 2-step reconstruction
        drifter_cell_count = np.count_nonzero(drifter_grid.grid.numobs5)

        if drifter_cell_count > drifter_threshold:
            # Do some interpolation stuff here
            kernel = io.Kernel(varym, 1300.0, 1.5)
            interp1 = io.OKInterpolator(drifter_grid.grid, kernel)
            interp1.make_covariance()
            for n in range(1, 4):
                for m in range(-1 * n, n):
                    interp1.add_spherical_harmonics_to_covariance(n, m, 0.2)

            interpolated_grid1 = interp1.do_interpolation()
            interpolated_grid1.data5[np.isnan(sampling_unc.sst.values[0:1, :, :])] = np.nan

            intermediate_ship_grid = ship_grid.grid - interpolated_grid1

            kernel = io.Kernel(varym, 1300.0, 1.5)
            interp2 = io.GPInterpolator(intermediate_ship_grid, kernel) # GP for second step as we have zero mean

            interp2.replace_covariance(interp1.posterior)
            interpolated_grid2 = interp2.do_interpolation()
            interpolated_grid2.data5[np.isnan(sampling_unc.sst.values[0:1, :, :])] = np.nan

            interpolated_grid2 = interpolated_grid2 + interpolated_grid1

        else:
            kernel = io.Kernel(varym, 1300.0, 1.5)
            interp2 = io.OKInterpolator(ship_grid.grid, kernel)
            interp2.make_covariance()
            for n in range(1, 4):
                for m in range(-1 * n, n):
                    interp2.add_spherical_harmonics_to_covariance(n, m, 0.2)

            interpolated_grid2 = interp2.do_interpolation()
            interpolated_grid2.data5[np.isnan(sampling_unc.sst.values[0:1, :, :])] = np.nan

        row = row + interpolated_grid2.calculate_regional_averages(regions, areas)
        row = row + interpolated_grid_ok.calculate_regional_averages(regions, areas)

        plot_more_up(
            [basic_grid.grid, basic_grid.grid, basic_grid.grid, interpolated_grid, biases, deck_biases,
             spheroidal, flat, local],
            [f'{year}-{month:02d} Basic grid', 'uncertainty', 'numobs', 'Interpolated grid',
             'Individual ship biases', 'Deck biases', 'Spherical Harmonics', 'Global mean',
             'Local', ],
            ['anom', 'unc', 'numobs', 'anom', 'anom', 'anom', 'anom', 'anom', 'anom'],
            data_dir / "ICOADS" / "Figures" / f"four_up_{year}{month:02d}.png"
        )

        time_series.loc[count] = row
        time_series.to_csv(data_dir / "ICOADS" / "OutputData" / f"timeseries_with_uncertainty_{year}{month:02d}.csv")

        # Transfer the data to xarray DataArrays and write out
        date_range = pd.date_range(start=f'{year}-{month:02d}-01', freq='1MS', periods=1)
        tdir = data_dir / "ICOADS" / "OutputData"
        tags = ['', '_ship', '_drifter', '_interp', '_interp_adjusted', '_interp_ok']
        for i, g in enumerate([
            basic_grid.grid,
            ship_grid.grid,
            drifter_grid.grid,
            interpolated_grid,
            interpolated_grid2,
            interpolated_grid_ok,
        ]):
            oo_anomalies = gridder.Grid.make_xarray(g.data5, res=5, times=date_range)
            oo_anomalies.to_netcdf(tdir / f"oo_anomalies{tags[i]}_{year}{month:02d}.nc")

            oo_uncertainty = gridder.Grid.make_xarray(g.unc5, res=5, times=date_range)
            oo_uncertainty.to_netcdf(tdir / f"oo_uncertainty{tags[i]}_{year}{month:02d}.nc")

            oo_numobs = gridder.Grid.make_xarray(g.numobs5, res=5, times=date_range)
            oo_numobs.to_netcdf(tdir / f"oo_numobs{tags[i]}_{year}{month:02d}.nc")
