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

import pytest
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import open_ocean.interpolation as io
import open_ocean.gridder as gridder


@pytest.fixture
def climatology():
    data_array = np.zeros((365, 180, 360))
    latitudes = np.linspace(-89.5, 89.5, 180)
    longitudes = np.linspace(-179.5, 179.5, 360)
    times = pd.date_range(start=f'1981-01-01', freq='1D', periods=365)

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


@pytest.fixture
def empty_grid(climatology):
    platform_id = np.array(["SHIP", "DRIFTER", "DRIFTER"])
    platform_type = np.array([1, 2, 2])
    lat = np.array([-89.5, -89.5, 0.0])
    lon = np.array([-179.5, -179.5, 0.0])
    date = np.array(
        [
            datetime(2000, 1, 1),
            datetime(2000, 1, 1),
            datetime(2000, 1, 1)
        ]
    )
    values = np.array([1.0, 4.0, -4.0])

    grid = gridder.Grid(
        2005, 1, platform_id, lat, lon, date, values, platform_type, climatology
    )

    grid.add_uncertainties()
    grid.add_sampling_uncertainties()
    grid.do_one_step_5x5_gridding()
    grid.calculate_covariance()

    return grid


def test_kernel():
    k = io.Kernel(1.0, 5.0, 0.5)

    x1 = np.arange(0.0, 50.0, 0.01)
    y1 = np.zeros(len(x1))
    z1 = np.zeros(len(x1))
    x2 = np.zeros(len(x1))
    y2 = np.zeros(len(x1))
    z2 = np.zeros(len(x1))

    k.get_covariances(x1, y1, z1, x2, y2, z2)

    assert k.C[0] == 1.0


def test_convert_lat_lon_to_euclidean():
    lat = np.array([0.0, 0.0, 90.0])
    lon = np.array([0.0, 90.0, 0.0])

    x, y, z = io.GPInterpolator.convert_lat_lon_to_euclidean(lat, lon)

    assert np.allclose(x, np.array([6371, 0.0, 0.0]))
    assert np.allclose(y, np.array([0.0, 6371, 0.0]))
    assert np.allclose(z, np.array([0.0, 0.0, 6371]))


def test_make_covariance(empty_grid):
    kernel = io.Kernel(0.6, 1000.0, 0.5)
    interp = io.GPInterpolator(empty_grid, kernel)

    interp.make_covariance()

    print(interp.cov)

def test_interp(empty_grid):
    kernel = io.Kernel(0.6, 1300.0, 1.5)
    interp = io.GPInterpolator(empty_grid, kernel)
    interp.do_interpolation()