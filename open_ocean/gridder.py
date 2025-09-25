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

import xarray as xr
import pandas as pd
import numpy as np
from scipy.stats import trim_mean
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def trim_mean_wrapper(inarr):
    return trim_mean(inarr, proportiontocut=0.1)


class Grid:

    def __init__(self, year, month, platform_id, lat, lon, date, value, platform_type, climatology):
        self.year = year
        self.month = month

        # Calculate anomalies
        self.x_index = self.get_x_index(lon)
        self.y_index = self.get_y_index(lat)
        self.t_index = self.get_t_index(date)
        self.value = value
        self.anomalies = self.calculate_anomalies(climatology)

        # Eliminate any nans in the anomalies that arise from climatology coverage
        non_missing = ~np.isnan(self.anomalies)

        self.x_index = self.x_index[non_missing]
        self.y_index = self.y_index[non_missing]
        self.t_index = self.t_index[non_missing]
        self.value = self.value[non_missing]
        self.anomalies = self.anomalies[non_missing]

        self.lat = lat[non_missing]
        self.lon = lon[non_missing]

        self.date = [d for present, d in zip(non_missing, date) if present]

        self.id = platform_id[non_missing]
        self.type = platform_type[non_missing]

        # Fill the uncertainties
        self.sigma_m = np.zeros(len(self.lat))
        self.sigma_b = np.zeros(len(self.lat))
        self.sigma_s = None
        self.add_uncertainties()

        # Initialise some attributes that will be filled later by the gridding methods
        self.xindex5 = (self.x_index / 5).astype(int)
        self.yindex5 = (self.y_index / 5).astype(int)
        self.xy1 = self.t_index * 1000000 + self.x_index * 1000 + self.y_index
        self.xy5 = self.xindex5 + self.yindex5 * 72

        self.weights = None
        self.weights5 = None

        self.data = None
        self.numobs = None
        self.unc = None

        self.data5 = None
        self.numobs5 = None
        self.numsobs5 = None
        self.unc5 = None

        self.covariance = None

    def add_sampling_uncertainties(self, sampling_unc):
        self.sigma_s = sampling_unc.sst.values[self.month - 1, :, :]
        self.sigma_s[np.isnan(self.sigma_s)] = 1.5

    def add_uncertainties(self, uncertainties=None):
        if uncertainties is None:
            uncertainties = [
                [1, 0.74, 0.71],  # Ships
                [2, 0.26, 0.29],  # Drifters
                [6, 0.26, 0.29],  # Drifters
                [3, 0.3, 0.2],  # Moorings
                [4, 0.3, 0.2],  # Moorings
                [5, 0.1, 0.05],  # Argo
            ]

        # Assign uncertainties for different observation types
        # Default value is same as for ships.
        self.sigma_m = np.zeros(len(self.lat)) + 0.74
        self.sigma_b = np.zeros(len(self.lat)) + 0.71

        for u in uncertainties:
            self.sigma_m[self.type == u[0]] = u[1]
            self.sigma_b[self.type == u[0]] = u[2]

    def do_1x1_gridding(self):
        df = pd.DataFrame(
            {
                "xy1": self.xy1,
                "value": self.anomalies,
                "x": self.x_index,
                "y": self.y_index,
                "t": self.t_index,
            }
        )

        grouped = df.groupby("xy1")

        means = grouped["value"].mean().values
        nobs = grouped["x"].count().values
        x = grouped["x"].first().values
        y = grouped["y"].first().values
        t = grouped["t"].first().values

        self.data = np.full((365, 180, 360), np.nan)
        self.numobs = np.zeros((365, 180, 360))

        if len(t) != 0:
            self.data[t, y, x] = means[:]
            self.numobs[t, y, x] = nobs[:]

    def do_two_step_5x5_gridding(self):
        # Build a dataframe for the 1x1 aggregation and then aggregate
        df1 = pd.DataFrame(
            {
                "xy1": self.xy1,
                "xy5": self.xy5,
                "value": self.anomalies,
                "x": self.x_index,
                "y": self.y_index,
                "t": self.t_index,
                "x5": self.xindex5,
                "y5": self.yindex5,
            }
        )

        grouped1 = df1.groupby("xy1")

        trimmed_means = grouped1["value"].agg(trim_mean_wrapper).values
        nobs = grouped1["x"].count().values
        x5 = grouped1["x5"].first().values
        y5 = grouped1["y5"].first().values
        xy5 = grouped1["xy5"].first().values

        xy1b = grouped1["xy1"].first().values
        weightb = 1 / nobs

        df1_match = pd.DataFrame(
            {
                "xy1": xy1b,
                "weightb": weightb,
            }
        )

        # Build another dataframe for 5x5 aggregation and then aggregate
        df5 = pd.DataFrame(
            {
                "xy5": xy5,
                "x5": x5,
                "y5": y5,
                "value": trimmed_means,
                "nobs": nobs
            }
        )

        grouped5 = df5.groupby("xy5")

        second_mean = grouped5["value"].agg(trim_mean_wrapper).values
        nobs = grouped5["nobs"].sum().values
        snobs = grouped5["nobs"].count().values
        x5 = grouped5["x5"].first().values
        y5 = grouped5["y5"].first().values

        xy5b = grouped5["xy5"].first().values
        weightc = 1 / snobs

        df2_match = pd.DataFrame(
            {
                "xy5": xy5b,
                "weightc": weightc,
            }
        )

        # Assign weights to the original observations
        df1 = pd.merge(df1, df1_match, on="xy1", how="left")
        df1 = pd.merge(df1, df2_match, on="xy5", how="left")
        self.weights5 = df1.weightb.values * df1.weightc.values

        # Make a grid and copy the grid cell averages into the grid
        self.data5 = np.full((1, 36, 72), np.nan)
        self.numobs5 = np.zeros((1, 36, 72))
        self.numsobs5 = np.zeros((1, 36, 72))
        self.unc5 = np.full((1, 36, 72), np.nan)

        self.data5[0, y5, x5] = second_mean[:]
        self.numobs5[0, y5, x5] = nobs[:]
        self.numsobs5[0, y5, x5] = snobs[:]

    def do_one_step_5x5_gridding(self):
        # Pack the data into a DataFrame so that we can use Pandas aggregation magic
        df = pd.DataFrame(
            {
                'xy5': self.xy5,
                'x': self.xindex5,
                'y': self.yindex5,
                'value': self.anomalies,
                'id': self.id,
                'sigma_m': self.sigma_m,
                'sigma_b': self.sigma_b,
            }
        )

        # Calculate the mean, number of observations and the indices of each grid cell with data in it.
        grouped = df.groupby("xy5")
        means = grouped["value"].mean().values
        nobs = grouped["x"].count().values
        x = grouped["x"].first().values
        y = grouped["y"].first().values
        xy5_unique = grouped["xy5"].first().values

        # Merge the weights back into the original dataframe
        df_match = pd.DataFrame(
            {
                "xy5": xy5_unique,
                "weight": 1 / nobs,
            }
        )
        df = pd.merge(df, df_match, on="xy5", how="left")
        self.weights5 = df.weight.values

        # Make a grid and copy the grid cell averages into the grid
        self.data5 = np.full((1, 36, 72), np.nan)
        self.numobs5 = np.zeros((1, 36, 72))
        self.numsobs5 = np.zeros((1, 36, 72))
        self.unc5 = np.full((1, 36, 72), np.nan)

        self.data5[0, y, x] = means[:]
        self.numobs5[0, y, x] = nobs[:]
        self.numsobs5[0, y, x] = nobs[:]

    def do_one_step_5x5_sampler_gridding(self, n_samples=1, rng=np.random.default_rng()):
        # Pack the data into a DataFrame so that we can use Pandas aggregation magic
        df = pd.DataFrame(
            {
                'xy5': self.xy5,
                'x': self.xindex5,
                'y': self.yindex5,
                'value': self.anomalies,
                'id': self.id,
                'sigma_m': self.sigma_m,
                'sigma_b': self.sigma_b,
            }
        )

        def subsample_mean(inarr):
            if n_samples > len(inarr):
                return np.nan
            return np.mean(rng.choice(inarr, size=n_samples, replace=False))

        # Calculate the mean, number of observations and the indices of each grid cell with data in it.
        grouped = df.groupby("xy5")
        means = grouped["value"].agg(subsample_mean).values
        nobs = grouped["x"].count().values
        x = grouped["x"].first().values
        y = grouped["y"].first().values
        xy5_unique = grouped["xy5"].first().values

        # Merge the weights back into the original dataframe
        df_match = pd.DataFrame(
            {
                "xy5": xy5_unique,
                "weight": 1 / nobs,
            }
        )
        df = pd.merge(df, df_match, on="xy5", how="left")
        self.weights5 = df.weight.values

        # Make a grid and copy the grid cell averages into the grid
        self.data5 = np.full((1, 36, 72), np.nan)
        self.numobs5 = np.zeros((1, 36, 72))
        self.numsobs5 = np.zeros((1, 36, 72))
        self.unc5 = np.full((1, 36, 72), np.nan)

        nobs[nobs > n_samples] = n_samples

        self.data5[0, y, x] = means[:]
        self.numobs5[0, y, x] = nobs[:]
        self.numsobs5[0, y, x] = nobs[:]

    def calculate_covariance(self):
        if self.weights5 is None:
            raise RuntimeError("No gridding weights. Please run a gridder first")

        df = pd.DataFrame(
            {
                'xy5': self.xy5,
                'x': self.xindex5,
                'y': self.yindex5,
                'weight5': self.weights5,
                'id': self.id,
                'sigma_m': self.sigma_m,
                'sigma_b': self.sigma_b,
            }
        )

        # Group the data by ID and grid cell
        groups = df.groupby(['id', 'xy5'])

        aggregated_groups = groups.agg(
            {
                'xy5': 'first',
                'x': 'first',
                'y': 'first',
                'weight5': 'sum',
                'sigma_m': 'first',
                'sigma_b': 'first',
            }
        )

        groups2 = aggregated_groups.groupby(['id'])

        self.covariance = np.zeros((2592, 2592))

        # loop over the IDs and for each ID calculate the contribution to the covariance matrix and add it on.
        for thisid, group in groups2:
            # Calculate the bits that we need to make the covariance
            weight_sigma_m_sq = np.power(group['weight5'].values * group['sigma_m'].values, 2)
            weight_sigma_b = group['weight5'].values * group['sigma_b'].values

            # The error covariance matrix for this ship is the outer product of the weight time sigma_b
            matrix = np.outer(weight_sigma_b, weight_sigma_b)
            # On the diagonal we need to add the uncorrelated part of the uncertainty.
            n = len(weight_sigma_b)
            matrix[np.diag_indices(n)] = matrix[np.diag_indices(n)] + weight_sigma_m_sq

            # Use the indices to locate this ID's contribution to the overall covariance matrix and add it on.
            selection = np.ix_(group['xy5'].values, group['xy5'].values)
            self.covariance[selection] = self.covariance[selection] + matrix[:, :]

        # Sampling uncertainty
        flat_numsobs = self.numsobs5.flatten()
        flat_numsobs[flat_numsobs != 0] = 1.0/flat_numsobs[flat_numsobs != 0]
        sampling_unc = (self.sigma_s.flatten()**2) * flat_numsobs
        # Add sampling uncertainty to the diagonal
        self.covariance[np.diag_indices(2592)] = self.covariance[np.diag_indices(2592)] + sampling_unc

        # Extract the diagonal of the covariance matrix and populate the uncertainty grid
        self.unc5[:, :, :] = np.sqrt((self.covariance[np.diag_indices(2592)]).reshape((1, 36, 72)))
        self.unc5[self.unc5 == 0] = np.nan

    def calculate_anomalies(self, climatology):
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
        if len(self.value) < 1:
            return np.array([])
        clim_values = climatology.sst.values[self.t_index, self.y_index, self.x_index]
        return self.value - clim_values

    def get_x_index(self, lon):
        """Calculate the x index from the input longitudes for the input climatology."""
        index = (lon + 180).astype(int)
        index[index > 359] = 0
        return index

    def get_y_index(self, lat):
        """Calculate the y index from the input latitudes for the input climatology."""
        index = (lat + 90).astype(int)
        index[index > 179] = 179
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
        if len(date) < 1:
            return np.array([])

        month = np.array([d.month for d in date])
        day = np.array([d.day for d in date])

        cumulative_month_lengths = np.array(
            [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
        )
        day_number = cumulative_month_lengths[month - 1] + day - 1
        return day_number.astype(int)

    def calculate_area_average_with_covariance(self, areas=None, lat_range=None, lon_range=None):
        """
        Calculate the area average of the grid and the corresponding uncertainty using the covariance. Grid cells
        are weighted by the cosine of the latitude. An additional keyword argument can be used to provide the
        areas of the grid cells to consider in the calculation.

        Parameters
        ----------
        areas: np.ndarray or None
            Array containing the area of each grid box
        lat_range: list or None
            Range of latitudes to consider in the calculation
        lon_range: list or None
            Range of longitudes to consider in the calculation. Longitudes must be specified in the range -180 to 180.
            If the first longitude is lower than the second then the average will be calculated between the two. If
            the first longitude is higher than the second then the average will be calculated from the first to 180
            degrees and from -180 degrees to the second. This is used to calculate area averages that cross the
            dateline.

        Returns
        -------
        float, float
            Area average and uncertainty of the average
        """
        if lat_range is not None:
            if lat_range[0] > lat_range[1]:
                raise ValueError("First element of latitude selection must be less than the second")

        # Use a deepcopy here otherwise the step where missing data are set to zero changes the original grid
        ds = Grid.make_xarray(copy.deepcopy(self.data5), res=5)
        weights = np.cos(np.deg2rad(ds.latitude)).values
        weights = np.repeat(np.reshape(weights, (1, 36, 1)), 72, axis=2)

        #    mask = np.full(weights.shape, False)
        if lat_range is not None:
            latitudes = np.repeat(np.reshape(ds.latitude.values, (1, 36, 1)), 72, axis=2)
            mask = (latitudes < lat_range[0]) | (latitudes > lat_range[1])
            weights[mask] = 0.0

        if lon_range is not None:
            longitudes = np.repeat(np.reshape(ds.longitude.values, (1, 1, 72)), 36, axis=1)
            if lon_range[0] < lon_range[1]:
                mask = (longitudes < lon_range[0]) | (longitudes > lon_range[1])
                weights[mask] = 0.0
            else:
                mask = (longitudes > lon_range[0]) & (longitudes < lon_range[1])
                weights[mask] = 0.0

        if areas is not None:
            weights[0, :, :] = weights[0, :, :] * areas[:, :]

        non_missing = ~np.isnan(ds.sst.values)
        weight_sum = np.sum(weights[non_missing])
        if weight_sum != 0:
            average = np.sum(ds.sst.values[non_missing] * weights[non_missing]) / weight_sum
        else:
            return np.nan, np.nan

        weights = np.reshape(weights, (1, 2592))
        data_array = np.reshape(ds.sst.values, (1, 2592))

        weights[np.isnan(data_array)] = 0
        data_array[np.isnan(data_array)] = 0

        unc_sq = np.matmul(np.matmul(weights, self.covariance), weights.transpose())
        weight_sum = np.sum(weights)
        unc = np.sqrt(unc_sq) / weight_sum

        return average, unc.item()

    def calculate_area_average(self):
        """Calculate aree average from the input latitude and longitude ranges"""
        ds = Grid.make_xarray(self.data5, res=5)
        weights = np.cos(np.deg2rad(ds.latitude))
        weighted_mean = ds.weighted(weights).mean(("longitude", "latitude"))
        return weighted_mean.sst.values[0]

    @staticmethod
    def make_xarray(data_array, res=5, times=None):
        if res == 5:
            latitudes = np.linspace(-87.5, 87.5, 36)
            longitudes = np.linspace(-177.5, 177.5, 72)
        elif res == 1:
            latitudes = np.linspace(-89.5, 89.5, 180)
            longitudes = np.linspace(-179.5, 179.5, 360)
        else:
            raise ValueError("Invalid res")

        ntime = data_array.shape[0]

        if times is None:
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

    def plot_covariance(self, filename=None):
        # Let's plot the covariance matrix (zoom in, it's pretty).
        plt.pcolormesh(self.covariance)
        plt.title("Covariance")
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)

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
    def plot_generic_map(ds, levels, filename=None):
        plt.figure()
        plt.gcf().set_size_inches(16, 9)
        proj = ccrs.PlateCarree()
        p = ds.sst.plot(
            transform=proj,
            subplot_kws={'projection': proj},
            levels=levels,
        )
        p.axes.coastlines()
        plt.title("")
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, )
        plt.close('all')

    def plot_map_1x1(self, filename=None):
        """Plot the grid as a map"""
        ds = Grid.make_xarray(self.data, res=1)
        ds = ds.mean(dim='time')
        Grid.plot_generic_map(ds, np.arange(-3, 3, 0.2), filename=filename)

    def plot_map_5x5(self, filename=None):
        """Plot the 5x5 grid as a map"""
        ds = Grid.make_xarray(self.data5, res=5)
        Grid.plot_generic_map(ds, np.arange(-3, 3, 0.2), filename=filename)

    def plot_map_numobs_5x5(self, filename=None):
        """Plot the 5x5 grid as a map"""
        ds = Grid.make_xarray(self.numobs5, res=5)
        Grid.plot_generic_map(ds, [0, 1, 2, 3, 4, 5, 10, 100], filename=filename)

    def plot_map_unc_5x5(self, filename=None, levels=None):
        """Plot a map of the uncertainties at 5x5 resolution"""
        if levels is None:
            levels = np.arange(0, 1.5, 0.1)
        ds = Grid.make_xarray(self.unc5, res=5)
        Grid.plot_generic_map(ds, levels, filename=filename)
