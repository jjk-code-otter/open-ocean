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
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma, kv


class Kernel:

    def __init__(self, variance, length_scale, shape):
        self.variance = variance
        self.length_scale = length_scale
        self.shape = shape

    def get_covariances(self, x1, y1, z1, x2, y2, z2):
        distances = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

        gamma_nu = gamma(self.shape)
        root2nu = np.sqrt(2 * self.shape) * distances / self.length_scale
        mb2k =  kv(self.shape, root2nu)

        C = (
                (self.variance ** 2) *
                ((2 ** (1 - self.shape)) / gamma_nu) *
                (root2nu ** self.shape) *
                mb2k
        )

        C[distances==0] = self.variance**2
        return C

class GPInterpolator:

    def __init__(self, grid, kernel):
        self.kernel = kernel
        self.grid = grid
        self.cov = None
        self.posterior = None

    @staticmethod
    def convert_lat_lon_to_euclidean(lat, lon):
        earth_radius = 6371
        lat = lat * np.pi / 180
        lon = lon * np.pi / 180

        x = earth_radius * np.cos(lat) * np.cos(lon)
        y = earth_radius * np.cos(lat) * np.sin(lon)
        z = earth_radius * np.sin(lat)

        return x, y, z

    def make_covariance(self, constant=None):
        latitudes = self.grid.get_latitudes().flatten()
        longitudes = self.grid.get_longitudes().flatten()

        x, y, z = self.convert_lat_lon_to_euclidean(latitudes, longitudes)

        z = z * 3.0

        x = np.repeat(np.reshape(x, (len(x), 1)), len(x), 1)
        y = np.repeat(np.reshape(y, (len(y), 1)), len(y), 1)
        z = np.repeat(np.reshape(z, (len(z), 1)), len(z), 1)

        self.cov = self.kernel.get_covariances(
            x, y, z,
            x.transpose(), y.transpose(), z.transpose()
        )

        if constant is not None:
            self.cov += constant**2

    def get_h(self):
        obs = self.grid.data5.flatten()
        nonmissing = ~np.isnan(obs)
        h = np.identity(len(obs))
        h = h[nonmissing, :]
        obs[~nonmissing] = 0.0
        obs = np.reshape(obs, (len(obs), 1))
        return h, obs

    def do_interpolation(self):
        if self.cov is None:
            self.make_covariance()

        # Get the observation selector matrix
        h, obs = self.get_h()

        cht = np.matmul(self.cov, h.transpose())
        hch = np.matmul(h, cht)

        # Get the observation error covariance at obs locations
        r = np.matmul(np.matmul(h, self.grid.covariance), h.transpose())

        inv_part = np.linalg.inv(hch + r)

        hobs = np.matmul(h, obs)
        mu = np.matmul(cht, np.matmul(inv_part, hobs))

        p = np.matmul(inv_part, cht.transpose())
        p = np.matmul(cht, p)
        p = self.cov - p
        self.posterior = p

        out_grid = copy.deepcopy(self.grid)
        out_grid.data5[0, :, :] = np.reshape(mu, (36, 72))
        out_grid.covariance = p

        out_grid.unc5[0, :, :] = np.reshape(p[np.diag_indices(2592)], (36,72))

        return out_grid
