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
from datetime import datetime
from itertools import count

import pytest

import numpy as np
import pandas as pd
import xarray as xr

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
def simple_grid(climatology):
    platform_id = np.array(["SHIP", "DRIFTER"])
    platform_type = np.array([1, 2])
    lat = np.array([-89.5, -89.5])
    lon = np.array([-179.5, -179.5])
    date = np.array([datetime(2000, 1, 1), datetime(2000, 1, 1)])
    values = np.array([1.0, 1.0])

    grid = gridder.Grid(2000, 1, platform_id, lat, lon, date, values, platform_type, climatology)

    return grid

def test_initialise_grid(simple_grid):
    assert np.all(simple_grid.value == 1.0)

    assert np.all(simple_grid.xindex5 == 0)
    assert np.all(simple_grid.yindex5 == 0)

    assert np.all(simple_grid.x_index == 0)
    assert np.all(simple_grid.y_index == 0)
    assert np.all(simple_grid.t_index ==  0)

    assert simple_grid.sigma_m[0] == 0.74
    assert simple_grid.sigma_b[0] == 0.71
    assert simple_grid.sigma_m[1] == 0.26
    assert simple_grid.sigma_b[1] == 0.29

def test_add_uncertainties(simple_grid):
    simple_grid.add_uncertainties(uncertainties=[])
    assert simple_grid.sigma_m[0] == 0.74
    assert simple_grid.sigma_b[0] == 0.71
    assert simple_grid.sigma_m[1] == 0.74
    assert simple_grid.sigma_b[1] == 0.71

    simple_grid.add_uncertainties(uncertainties=[[1, 2, 3],[2, 3, 4]])
    assert simple_grid.sigma_m[0] == 2
    assert simple_grid.sigma_b[0] == 3
    assert simple_grid.sigma_m[1] == 3
    assert simple_grid.sigma_b[1] == 4

def test_do_1x1_gridding(simple_grid):
    simple_grid.do_1x1_gridding()

    assert np.count_nonzero(np.isnan(simple_grid.data)) == 365*180*360-1
    assert simple_grid.data[0,0,0] == 1.0
    assert simple_grid.numobs[0,0,0] == 2

def test_do_two_step_5x5_gridder(simple_grid):
    simple_grid.do_two_step_5x5_gridding()

    assert np.count_nonzero(np.isnan(simple_grid.data5)) == 72*36-1
    assert np.sum(simple_grid.numobs5) == 2
    assert simple_grid.data5[0,0,0] == 1.0
    assert simple_grid.numobs5[0,0,0] == 2
    assert np.all(simple_grid.weights5 == 0.5)

def test_get_x_index(simple_grid):
    result = simple_grid.get_x_index(np.array([-179.5, 179.5, -0.5, 0.5]))
    assert np.all(result == np.array([0, 359, 179, 180]))

    # Grid cell boundaries are bumped east.
    result = simple_grid.get_x_index(np.array([-180, 180, 179.99999, 0.0, 3.0]))
    assert np.all(result == np.array([0, 0, 359, 180, 183]))

def test_get_y_index(simple_grid):
    result = simple_grid.get_y_index(np.array([-89.5, 89.5, -0.5, 0.5]))
    assert np.all(result == np.array([0, 179, 89, 90]))

    # Grid cell boundaries are bumped north.
    result = simple_grid.get_y_index(np.array([-90, 90, 0.0, 5.0]))
    assert np.all(result == np.array([0, 179, 90, 95]))

def test_get_t_index(simple_grid):
    result = simple_grid.get_t_index(pd.date_range("2001-01-01", periods=365, freq="D"))
    np.all(result == np.linspace(0, 364, 365))