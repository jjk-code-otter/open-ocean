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

import numpy as np

import open_ocean.vbpca as vb

def test_initialise():
    """This isn't really a test as such"""
    rng = np.random.default_rng()

    selector = rng.uniform(0,1,(100,5,10))

    data = rng.normal(0.0, 1.0, (100, 5, 10))
    unc = np.zeros((100, 5, 10)) + 0.1

    data[selector > 0.95] = np.nan
    unc[selector > 0.95] = np.nan

    interpolator = vb.VBPCA(data, unc, 2)

    n_iterations = 50

    for i in range(n_iterations):
        interpolator.project_space()
        interpolator.project_time()
        interpolator.update_v()

    recon = interpolator.make_recon()

