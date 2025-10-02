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
import numpy as np
from scipy.optimize import minimize_scalar


class VBPCA:
    """
    A class for performing Variational Bayesian Principal Component Analysis (VBPCA).
    """

    def __init__(
            self,
            data: np.ndarray,
            unc: np.ndarray,
            n_eofs: int,
            mask_percentage: float = 0.333,
            grid_areas: np.ndarray = None
    ):
        """
        Create the basic interpolator by feeding it data.

        Parameters
        ----------
        data: np.ndarray
            An arrray containing the data to be interpolated. It expects an array which is (time x latitude x longitude)
        unc: np.ndarray
            An array containing the uncertainty of the data. It expects an array which is (time x latitude x longitude)
        n_eofs: int
            Number of patterns to estimate
        mask_percentage: float
            Fraction of time points that must be populated to be included in the final fields.
        grid_areas: np.ndarray or None
            (optional) An array containing the area of the grid cells to be interpolated.
        """
        # Grid areas are assumed to be 1.0 unless otherwise stated.
        self.grid_areas = grid_areas
        if self.grid_areas is not None:
            self.grid_areas = self.grid_areas / np.max(self.grid_areas)
            self.grid_areas = np.reshape(self.grid_areas, (1, data.shape[1], data.shape[2]))
            self.grid_areas = np.repeat(self.grid_areas, data.shape[0], axis=0)
        else:
            self.grid_areas = np.zeros_like(data) + 1.0

        # Set up the random number generator
        rng = np.random.default_rng(222)

        n_time = data.shape[0]

        # build the data mask and set the number of spatial points
        mask = np.where(np.isnan(data), 0, 1)
        mask = np.sum(mask, axis=0) / n_time
        mask = mask > mask_percentage
        self.mask = mask
        n_space = np.count_nonzero(mask)

        self.input_data = data

        self.data = data * grid_areas
        self.data = self.data[:, mask]
        self.data = np.reshape(self.data, (n_time, n_space))

        self.unc = unc * grid_areas
        self.unc = self.unc[:, mask] ** 2
        self.unc = np.reshape(self.unc, (n_time, n_space))

        self.w = rng.normal(0.0, 1.0, (n_eofs, n_space))
        self.x = rng.normal(0.0, 1.0, (n_eofs, n_time))

        self.full_recon = np.matmul(self.x.transpose(), self.w)

        self.v = 1.0
        self.mu = np.zeros(n_space)

        self.alpha0 = 1.0
        self.alphas = np.zeros(n_eofs) + 1.0

        self.n_eofs = n_eofs
        self.n_space = n_space
        self.n_time = n_time

    def project_space(self) -> None:
        """
        The patterns are estimated by alternately updating the space and time parts of the model. This method
        runs a space update.

        Returns
        -------
        None
        """
        for t in range(self.n_time):
            data_vector = np.reshape(self.data[t, :] - self.mu[:], (self.n_space, 1))
            unc_vector = np.reshape(self.unc[t, :], (self.n_space, 1))

            nonmissing = ~np.isnan(data_vector)
            n = np.count_nonzero(nonmissing)

            h = np.identity(self.n_space)
            h = h[nonmissing[:, 0], :]

            inv_unc = 1 / (unc_vector[nonmissing[:, 0], 0] + self.v)

            wht = np.matmul(self.w, h.transpose())
            r = np.diag(inv_unc)
            whtr = np.matmul(wht, r)

            sigma_xt = np.identity(self.n_eofs) + np.matmul(whtr, wht.transpose())
            sigma_xt = np.linalg.inv(sigma_xt)

            #            xt = np.matmul(h, data_vector)
            xt = np.matmul(whtr, data_vector[nonmissing[:, 0], :])

            xt = np.matmul(sigma_xt, xt)

            self.x[:, t] = xt[:, 0]

    def project_time(self):
        """
        The patterns are estimated by alternately updating the space and time parts of the model. This method
        runs a time update.

        Returns
        -------
        None
        """
        self.full_recon = np.matmul(self.x.transpose(), self.w)
        residuals = (self.data - self.full_recon) / (self.unc + self.v)

        for i in range(self.n_space):
            unc_vector = np.reshape(self.unc[:, i], (self.n_time, 1))

            nonmissing = ~np.isnan(unc_vector)
            n = np.count_nonzero(nonmissing)

            mutilde = 1 / (1 / self.alpha0 + np.sum(1 / (unc_vector[nonmissing[:, 0], 0] + self.v)))
            self.mu[i] = mutilde * np.sum(residuals[nonmissing[:, 0], i])

            data_vector = np.reshape(self.data[:, i], (self.n_time, 1)) - self.mu[i]

            h = np.identity(self.n_time)
            h = h[nonmissing[:, 0], :]

            inv_unc = 1 / (unc_vector[nonmissing[:, 0], 0] + self.v)

            xht = np.matmul(self.x, h.transpose())
            r = np.diag(inv_unc)
            xhtr = np.matmul(xht, r)

            sigmawi = np.diag(1 / self.alphas) + np.matmul(xhtr, xht.transpose())
            sigmawi = np.linalg.inv(sigmawi)

            wi = np.matmul(xhtr, data_vector[nonmissing[:, 0], :])
            wi = np.matmul(sigmawi, wi)

            self.w[:, i] = wi[:, 0]

    def update_v(self) -> None:
        """
        After every time update, we also need to update the variance parameter. This method finds the variance
        parameter that minimizes the `func_to_minimize` function.

        Returns
        -------
        None
        """
        self.full_recon = np.matmul(self.x.transpose(), self.w)
        mu_repeated = np.repeat(np.reshape(self.mu, (1, self.n_space)), self.n_time, 0)
        residuals_sq = (self.data - mu_repeated - self.full_recon) ** 2
        nonmissing = ~np.isnan(self.data)

        def func_to_minimize(in_v):
            vit = self.unc + in_v
            cvb_vec = residuals_sq / vit + np.log(vit)
            return np.sum(cvb_vec[nonmissing])

        res = minimize_scalar(func_to_minimize, bounds=(0.0, 10.0), method='bounded')
        self.v = res.x

    def fit_model(self, max_iterations=100) -> None:
        """
        Fits the model to the data by successively running the project_space, project_time and update_v methods.
        The maximum number of iterations can be specified and defaults to 100. Convergence is detected when the
        variance is unchanged from one iteration to the next.

        Parameters
        ----------
        max_iterations: int
            Maximum number of iterations to run.

        Returns
        -------
        None
        """
        previous_v = 99.99
        for i in range(max_iterations):
            print(f"Iteration {i}")
            self.project_space()
            self.project_time()
            self.update_v()
            print(f'{previous_v:.5f}, {self.v:.5f}, {previous_v - self.v}')
            if self.v == previous_v:
                break
            else:
                previous_v = self.v

    def make_recon(self) -> np.ndarray:
        """
        Build the reconstruction using information currently stored in the VBPCA object.

        Returns
        -------
        np.ndarray
            Array containing the reconstruction of the data.
        """
        self.full_recon = np.matmul(self.x.transpose(), self.w)
        mu_repeated = np.repeat(np.reshape(self.mu, (1, self.n_space)), self.n_time, 0)

        grid = copy.deepcopy(self.input_data)
        grid[:, :, :] = np.nan
        grid[:, self.mask] = self.full_recon[:, :] + mu_repeated[:, :]
        grid = grid / self.grid_areas

        return grid

    def make_pc_series(self) -> np.ndarray:
        """
        Return the timeseries weights for each of the patterns

        Returns
        -------
        np.ndarray
            Array containing the timeseries weights for each pattern shape (n_eofs, n_time)
        """
        return self.x

    def make_eofs(self):
        """
        Return the patterns as a grid

        Returns
        -------
        np.ndarray
            Array containing the pattern weights for each pattern shape (n_eofs+2, 36, 72). The first n_eofs
            entries are the patterns and then the next two fields contain the mean field (mu) and the variance
            parameter (v).
        """
        grid = copy.deepcopy(self.input_data)
        grid[:, :, :] = np.nan
        grid[0:self.n_eofs, self.mask] = self.w[:, :]
        grid[self.n_eofs:, self.mask] = self.mu[:]
        grid[self.n_eofs + 1:, self.mask] = self.v

        return grid

    def recalc_as_eigenvectors(self):
        reconstituted_cov = np.matmul(self.w.transpose(), self.w)
        vals, vecs = np.linalg.eig(reconstituted_cov)
        grid = copy.deepcopy(self.input_data)
        grid[:, :, :] = np.nan
        grid[0:self.n_eofs, self.mask] = vecs[:, 0:self.n_eofs].transpose()
        return grid