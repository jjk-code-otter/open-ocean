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
from scipy.special import gamma, kv, sph_harm_y


def convert_lat_lon_to_euclidean(lat, lon):
    earth_radius = 6371
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180

    x = earth_radius * np.cos(lat) * np.cos(lon)
    y = earth_radius * np.cos(lat) * np.sin(lon)
    z = earth_radius * np.sin(lat)

    return x, y, z


class Kernel:

    def __init__(self, variance, length_scale, shape):
        self.variance = variance
        self.length_scale = length_scale
        self.shape = shape

    def get_covariances(self, x1, y1, z1, x2, y2, z2):
        distances = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

        gamma_nu = gamma(self.shape)
        root2nu = np.sqrt(2 * self.shape) * distances[distances != 0] / self.length_scale
        mb2k =  kv(self.shape, root2nu)

        C = np.full_like(distances, 0.0)

        if len(self.variance) == len(x1):

            covariances = np.outer(self.variance, self.variance)

            C[distances != 0] = (
                    covariances[distances != 0] *
                    ((2 ** (1 - self.shape)) / gamma_nu) *
                    (root2nu ** self.shape) *
                    mb2k
            )

        elif len(self.variance) == 1:
            C[distances != 0] = (
                    (self.variance ** 2) *
                    ((2 ** (1 - self.shape)) / gamma_nu) *
                    (root2nu ** self.shape) *
                    mb2k
            )
        else:
            raise RuntimeError(f"Bad variance length {len(self.variance)}")

        C[distances==0] = self.variance*self.variance
        return C

class KalmanInterpolator:

    def __init__(self):
        self.running_mean = None
        self.running_cov = None

    def update(self, grid, variances, innovations, constant, innovation_constant, autocorr):

        if self.running_mean is None:
            kernel = Kernel(variances, 1300.0, 1.5)
            interpolator = GPInterpolator(grid, kernel)
            interpolator.make_covariance(constant=constant)

            interpolated_grid = interpolator.do_interpolation()

            self.running_mean = interpolated_grid * autocorr
            ft = np.diag(autocorr.flatten())
            self.running_cov = np.matmul(np.matmul(ft, interpolator.posterior), ft.transpose())
            del ft

        else:
            kernel = Kernel(innovations, 1300.0, 1.5)
            update_grid = grid - self.running_mean
            interpolator = GPInterpolator(update_grid, kernel)
            interpolator.make_covariance(constant=innovation_constant)

            cov_update = self.running_cov + interpolator.cov
            interpolator.replace_covariance(cov_update)

            interpolated_grid = interpolator.do_interpolation()
            interpolated_grid = interpolated_grid + self.running_mean

            self.running_mean = interpolated_grid * autocorr
            ft = np.diag(autocorr.flatten())
            self.running_cov = np.matmul(np.matmul(ft, interpolator.posterior), ft.transpose())
            del ft

        return interpolated_grid


class OKInterpolator:

    def __init__(self, grid, kernel, basis=None):
        self.grid = grid
        self.kernel = kernel
        if basis is None:
            self.basis = np.full((1, 2592), 1.0)
        else:
            self.basis = np.full((1+len(basis), 2592), 1.0)
            for i in range(len(basis)):
                self.basis[i+1, :] = basis[i][:]

        self.cov = None
        self.posterior = None
        self.kriginv = None
        self.beta = None

    def replace_covariance(self, input_covariance):
        """Replace the current covariance with a completely new one"""
        self.cov = input_covariance

    def add_spherical_harmonics_to_covariance(self, n, m, variance):
        """
        Add spherical harmonics to the covariance matrix. This is the real component of the
        harmonics.

        Parameters
        ----------
        n: int
            n >= 0 Order of the harmonic.
        m: int
            in range -n, n
        variance: float
            variance for the harmonic.

        Returns
        -------

        """
        # Not that azimuth is defined from the north pole.
        latitudes = (90.0 - self.grid.get_latitudes().flatten()) * np.pi / 180.
        longitudes = self.grid.get_longitudes().flatten() * np.pi / 180.

        sph_cov = np.real(sph_harm_y(n, m, latitudes, longitudes))
        sph_cov = sph_cov / np.max(sph_cov)
        sph_cov = variance * sph_cov

        # Now convert into covariance matrix
        sph_cov = np.reshape(sph_cov, (2592, 1))
        sph_cov = np.outer(sph_cov, sph_cov)

        # Add to covariance.
        self.cov =  sph_cov + self.cov

        return sph_cov

    def make_covariance(self, constant=None):
        latitudes = self.grid.get_latitudes().flatten()
        longitudes = self.grid.get_longitudes().flatten()

        x, y, z = convert_lat_lon_to_euclidean(latitudes, longitudes)

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

        self.kriginv = np.linalg.inv(hch + r)

        hobs = np.matmul(h, obs)

        mu = np.matmul(cht, np.matmul(self.kriginv, hobs))

        bht = np.matmul(self.basis, h.transpose())
        bhtinv = np.matmul(bht, self.kriginv)

        g = self.basis - np.matmul(bhtinv, cht.transpose())
        beta = np.matmul(bhtinv, bht.transpose())
        betainv = np.linalg.inv(beta)
        beta = np.matmul(np.matmul(betainv, bhtinv), hobs)

        try:
            self.beta = beta.item()
        except:
            self.beta = beta

        mu = mu + np.matmul(g.transpose(), beta)

        p = np.matmul(self.kriginv, cht.transpose())
        p = np.matmul(cht, p)
        p = self.cov - p

        p = p + np.matmul(np.matmul(g.transpose(), betainv), g)

        self.posterior = p

        out_grid = copy.deepcopy(self.grid)
        out_grid.data5[0, :, :] = np.reshape(mu, (36, 72))
        out_grid.covariance = p

        out_grid.unc5[0, :, :] = np.reshape(p[np.diag_indices(2592)], (36,72))

        return out_grid

    def project_covariance(self, projection_covariance):
        # Get the observation selector matrix
        h, obs = self.get_h()

        cht = np.matmul(self.cov, h.transpose())
        hch = np.matmul(h, cht)

        # Get the observation error covariance at obs locations
        r = np.matmul(np.matmul(h, self.grid.covariance), h.transpose())
        pc = np.matmul(projection_covariance, h.transpose())

        if self.kriginv is None:
            self.kriginv = np.linalg.inv(hch + r)

        hobs = np.matmul(h, obs)
        mu = np.matmul(pc, np.matmul(self.kriginv, hobs))

        p = np.matmul(self.kriginv, pc.transpose())
        p = np.matmul(pc, p)
        p = projection_covariance - p

        out_grid = copy.deepcopy(self.grid)
        out_grid.data5[0, :, :] = np.reshape(mu, (36, 72))
        out_grid.covariance = p

        out_grid.unc5[0, :, :] = np.reshape(p[np.diag_indices(2592)], (36, 72))

        return out_grid

    def project_covariances_from_dict(self, cov_dict):
        out_grids = {}
        for key, values in cov_dict.items():
            built_cov = np.zeros((2592, 2592))
            selection = np.ix_(values[2], values[2])
            built_cov[selection] = built_cov[selection] + values[0][:, :]

            g = self.project_covariance(built_cov)

            # Need to scale by the inverse of the weights to get the expected constant value
            scaling = np.zeros(2592) * 1.0
            scaling[values[2]] = values[1][:]
            unc = np.sqrt(g.covariance[np.diag_indices(2592)] / scaling)

            scaling = np.reshape(scaling, (1, 36,72))
            g.data5 = g.data5 / scaling
            g.unc5 = np.reshape(unc, (1, 36,72))

            out_grids[key] = g

        return out_grids


class GPInterpolator:

    def __init__(self, grid, kernel):
        self.kernel = kernel
        self.grid = grid
        self.cov = None
        self.posterior = None
        self.kriginv = None


    def replace_covariance(self, input_covariance):
        """Replace the current covariance with a completely new one"""
        self.cov = input_covariance

    def add_spherical_harmonics_to_covariance(self, n, m, variance):
        """
        Add spherical harmonics to the covariance matrix. This is the real component of the
        harmonics.

        Parameters
        ----------
        n: int
            n >= 0 Order of the harmonic.
        m: int
            in range -n, n
        variance: float
            variance for the harmonic.

        Returns
        -------

        """
        # Not that azimuth is defined from the north pole.
        latitudes = (90.0 - self.grid.get_latitudes().flatten()) * np.pi / 180.
        longitudes = self.grid.get_longitudes().flatten() * np.pi / 180.

        sph_cov = np.real(sph_harm_y(n, m, latitudes, longitudes))
        sph_cov = sph_cov / np.max(sph_cov)
        sph_cov = variance * sph_cov

        # Now convert into covariance matrix
        sph_cov = np.reshape(sph_cov, (2592, 1))
        sph_cov = np.outer(sph_cov, sph_cov)

        # Add to covariance.
        self.cov =  sph_cov + self.cov

        return sph_cov

    def make_covariance(self, constant=None):
        latitudes = self.grid.get_latitudes().flatten()
        longitudes = self.grid.get_longitudes().flatten()

        x, y, z = convert_lat_lon_to_euclidean(latitudes, longitudes)

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

        self.kriginv = np.linalg.inv(hch + r)

        hobs = np.matmul(h, obs)
        mu = np.matmul(cht, np.matmul(self.kriginv, hobs))

        p = np.matmul(self.kriginv, cht.transpose())
        p = np.matmul(cht, p)
        p = self.cov - p
        self.posterior = p

        out_grid = copy.deepcopy(self.grid)
        out_grid.data5[0, :, :] = np.reshape(mu, (36, 72))
        out_grid.covariance = p

        out_grid.unc5[0, :, :] = np.reshape(p[np.diag_indices(2592)], (36,72))

        return out_grid

    def project_covariance(self, projection_covariance):
        # Get the observation selector matrix
        h, obs = self.get_h()

        cht = np.matmul(self.cov, h.transpose())
        hch = np.matmul(h, cht)

        # Get the observation error covariance at obs locations
        r = np.matmul(np.matmul(h, self.grid.covariance), h.transpose())
        pc = np.matmul(projection_covariance, h.transpose())

        if self.kriginv is None:
            self.kriginv = np.linalg.inv(hch + r)

        hobs = np.matmul(h, obs)
        mu = np.matmul(pc, np.matmul(self.kriginv, hobs))

        p = np.matmul(self.kriginv, pc.transpose())
        p = np.matmul(pc, p)
        p = projection_covariance - p

        out_grid = copy.deepcopy(self.grid)
        out_grid.data5[0, :, :] = np.reshape(mu, (36, 72))
        out_grid.covariance = p

        out_grid.unc5[0, :, :] = np.reshape(p[np.diag_indices(2592)], (36, 72))

        return out_grid

    def project_covariances_from_dict(self, cov_dict):
        out_grids = {}
        for key, values in cov_dict.items():
            built_cov = np.zeros((2592, 2592))
            selection = np.ix_(values[2], values[2])
            built_cov[selection] = built_cov[selection] + values[0][:, :]

            g = self.project_covariance(built_cov)

            # Need to scale by the inverse of the weights to get the expected constant value
            scaling = np.zeros(2592) * 1.0
            scaling[values[2]] = values[1][:]
            unc = np.sqrt(g.covariance[np.diag_indices(2592)] / scaling)

            scaling = np.reshape(scaling, (1, 36,72))
            g.data5 = g.data5 / scaling
            g.unc5 = np.reshape(unc, (1, 36,72))

            out_grids[key] = g

        return out_grids
