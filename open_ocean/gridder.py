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

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


class Grid:

    def __init__(self, year, month, id, lat, lon, date, value, type, climatology):
        self.year = year
        self.month = month
        self.lat = lat
        self.lon = lon
        self.date = date
        self.value = value
        self.id = id
        self.type = type

        # Assign uncertainties for different observation types
        # Ships
        self.sigma_m = np.zeros(len(self.lat)) + 0.74
        self.sigma_b = np.zeros(len(self.lat)) + 0.71

        # Drifters
        self.sigma_m[self.type == 2] = 0.26
        self.sigma_b[self.type == 2] = 0.29
        self.sigma_m[self.type == 6] = 0.26
        self.sigma_b[self.type == 6] = 0.29

        # Moorings
        self.sigma_m[self.type == 3] = 0.3
        self.sigma_b[self.type == 3] = 0.2
        self.sigma_m[self.type == 4] = 0.3
        self.sigma_b[self.type == 4] = 0.2

        # Argo
        self.sigma_m[self.type == 5] = 0.1
        self.sigma_b[self.type == 5] = 0.05

        self.x_index = self.get_x_index(self.lon)
        self.y_index = self.get_y_index(self.lat)
        self.t_index = self.get_t_index(self.date)

        self.data = np.empty((360, 180))

        self.anomalies = self.anomalize(climatology)

        self.do_gridding()

    def do_gridding(self):
        unique_index = self.t_index * 1000000 + self.x_index * 1000 + self.y_index

        df = pd.DataFrame(
            {
                "uid": unique_index,
                "value": self.anomalies,
                "x": self.x_index,
                "y": self.y_index,
                "t": self.t_index,
            }
        )

        means = df.groupby("uid")["value"].mean().values
        nobs = df.groupby("uid")["x"].count().values
        x = df.groupby("uid")["x"].first().values
        y = df.groupby("uid")["y"].first().values
        t = df.groupby("uid")["t"].first().values

        self.data = np.full((365, 180, 360), np.nan)
        self.nobs = np.zeros((365, 180, 360))

        self.data[t, y, x] = means[:]
        self.nobs[t, y, x] = nobs[:]

    def make_5x5_grid_with_covariance(self):
        # The indices in the 5x5 grid are simply related to the 1x1 grid already calculated
        xindex5 = (self.x_index / 5).astype(int)
        yindex5 = (self.y_index / 5).astype(int)
        # Need an index that uniquely identifies every one of the 2592 5x5 grid cells
        xy5 = xindex5 + yindex5 * 72

        # Pack the data into a DataFrame so that we can use Pandas magic
        df = pd.DataFrame(
            {
                'xy5': xy5,
                'x': xindex5,
                'y': yindex5,
                'value': self.anomalies,
                'id': self.id,
                'sigma_m': self.sigma_m,
                'sigma_b': self.sigma_b,
            }
        )

        # Calculate the mean, number of observations and the indices of each grid cell with data in it.
        means = df.groupby("xy5")["value"].mean().values
        nobs = df.groupby("xy5")["x"].count().values
        x = df.groupby("xy5")["x"].first().values
        y = df.groupby("xy5")["y"].first().values
        xy5_unique = df.groupby("xy5")["xy5"].first().values

        # Make a grid and copy the grid cell averages into the grid
        self.data5 = np.full((1, 36, 72), np.nan)
        self.nobs5 = np.zeros((1, 36, 72))
        self.unc = np.full((1, 36, 72), np.nan)

        self.data5[0, y, x] = means[:]
        self.nobs5[0, y, x] = nobs[:]
        nobs_flat = self.nobs5.flatten()  # We'll need this to calculate weights

        # Group the data by ID and grid cell
        groups = df.groupby(['id', 'xy5'])

        aggregated_groups = groups.agg(
            {
                'xy5': 'first',
                'x': 'first',
                'y': 'first',
                'value': 'count',
                'sigma_m': 'first',
                'sigma_b': 'first',
            }
        )

        groups2 = aggregated_groups.groupby(['id'])

        self.covariance = np.zeros((2592, 2592))

        # loop over the IDs and for each ID calculate the contribution to the covariance matrix and add it on.
        for thisid, group in groups2:
            # Using a simple aritmetic mean of each 5 degree grid cell
            weights = group['value'].values / nobs_flat[group['xy5'].values]

            # Calculate the bits that we need to make the covariance
            weight_sigma_m_sq = weights * weights * weights * group['sigma_m'].values * group['sigma_m'].values
            weight_sigma_b = weights * group['sigma_b'].values

            # The error covariance matrix for this ship is the outer product of the weight time sigma_b
            matrix = np.outer(weight_sigma_b, weight_sigma_b)
            # On the diagonal we need to add the uncorrelated part of the uncertainty.
            n = len(weight_sigma_b)
            matrix[np.diag_indices(n)] = matrix[np.diag_indices(n)] + weight_sigma_m_sq

            # Use the indices to locate this ID's contribution to the overall covariance matrix and add it on.
            selection = np.ix_(group['xy5'].values, group['xy5'].values)
            self.covariance[selection] = self.covariance[selection] + matrix[:, :]

        # Extract the diagonal of the covariance matrix and populate the uncertainty grid
        self.unc[:, :, :] = np.sqrt((self.covariance[np.diag_indices(2592)]).reshape((1, 36, 72)))
        self.unc[self.unc == 0] = np.nan

    def anomalize(self, climatology):
        """Calculate anomalies relative to the input climatology.

        Parameters
        ----------
        climatology: xarray.DataArray
            The climatology that will be used to calculate the anomalies.

        Returns
        -------
        np.ndarray
            Array containing the anomalies.
        """
        clim_values = climatology.sst.values[self.t_index, self.y_index, self.x_index]
        return self.value - clim_values

    def get_x_index(self, lon):
        """Calculate the x index from the input longitudes for the input climatology."""
        index = (lon + 180).astype(int)
        return index

    def get_y_index(self, lat):
        """Calculate the y index from the input latitudes for the input climatology."""
        index = (lat + 90).astype(int)
        return index

    def get_t_index(self, date):
        """Calculate the t index from the input date for the input climatology.

        Parameters
        ----------
        date: datetime.datetime
            array of datetime objects for which the time index will be calculated.

        Returns
        -------
        np.ndarray
            Array of time indices
        """
        month = np.array([d.month for d in date])
        day = np.array([d.day for d in date])

        cumulative_month_lengths = np.array(
            [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
        )
        day_number = cumulative_month_lengths[month - 1] + day - 1
        return day_number.astype(int)

    @staticmethod
    def make_xarray(data_array, res=5):
        if res == 5:
            latitudes = np.linspace(-87.5, 87.5, 36)
            longitudes = np.linspace(-177.5, 177.5, 72)
        elif res == 1:
            latitudes = np.linspace(-89.5, 89.5, 180)
            longitudes = np.linspace(-179.5, 179.5, 360)

        ntime = data_array.shape[0]

        times = pd.date_range(start=f'1851-01-01', freq='1D', periods=ntime)

        ds = xr.Dataset({
            'sst': xr.DataArray(
                data=data_array,
                dims=['time', 'latitude', 'longitude'],
                coords={'time': times, 'latitude': latitudes, 'longitude': longitudes},
                attrs={'long_name': 'sea surface temperature', 'units': 'K'}
            )
        },
            attrs={'project': 'NA'}
        )
        return ds

    def plot_covariance(self):
        # Let's plot the covariance matrix (zoom in, it's pretty).
        plt.pcolormesh(self.covariance)
        plt.title("Covariance")
        plt.show()

    def plot_covariance_row(self, lat, lon):
        """Plot a row from the covariance matrix based on specified latitude and longitude."""
        # Work out which xy grid cell the lat and lon point is in
        x = int(self.get_x_index(np.array([lon])) / 5)
        y = int(self.get_y_index(np.array([lat])) / 5)
        xy = x + y * 72
        # Extract the climatology row to the uncertainty grid
        unc_example = np.full((1, 36, 72), np.nan)
        unc_example[:, :, :] = np.sqrt((self.covariance[xy, :]).reshape((1, 36, 72)))
        unc_example[unc_example == 0] = np.nan
        # Plot the grid
        ds = Grid.make_xarray(unc_example, res=5)
        Grid.plot_generic_map(ds, np.arange(0, 0.2, 0.01))

    @staticmethod
    def plot_generic_map(ds, levels):
        plt.figure()
        proj = ccrs.PlateCarree()
        p = ds.sst.plot(
            transform=proj,
            subplot_kws={'projection': proj},
            levels=levels

        )
        p.axes.coastlines()
        plt.title("")
        plt.show()
        plt.close('all')

    def plot_map(self):
        """Plot the grid as a map"""
        ds = Grid.make_xarray(self.data, res=1)
        ds = ds.mean(dim='time')
        Grid.plot_generic_map(ds, np.arange(-3, 3, 0.2))

    def plot_map5(self):
        """Plot the 5x5 grid as a map"""
        ds = Grid.make_xarray(self.data5, res=5)
        Grid.plot_generic_map(ds, np.arange(-3, 3, 0.2))

    def plot_map_unc5(self):
        """Plot a map of the uncertainties at 5x5 resolution"""
        ds = Grid.make_xarray(self.unc, res=5)
        Grid.plot_generic_map(ds, np.arange(0, 1.5, 0.1))
