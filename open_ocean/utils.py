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
from itertools import product
from datetime import datetime
import numpy as np

def convert_climatology_to_ocean_areas(climatology):
    # Use the SST values in the climatology as an ocean mask. Set area to 1 for gridcells with SSTs, 0 otherwise
    ocean = np.where(np.isnan(climatology.sst[0].values), 0.0, 1.0)
    # Areas are the ocean area in the grid cell
    areas = np.zeros((36, 72))
    for xx, yy in product(range(72), range(36)):
        areas[yy, xx] = np.mean(ocean[yy * 5:(yy + 1) * 5, xx * 5:(xx + 1) * 5])
    return areas

def convert_dates(months, days):
    return [datetime(2020, months[i], days[i]) for i in range(len(months))]
