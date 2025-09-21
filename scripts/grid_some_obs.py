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

import xarray as xr
import numpy as np
from pathlib import Path
import os
from datetime import datetime

if __name__ == "__main__":
    data_dir = Path(os.getenv("DATADIR"))

    year = 1983
    month = 1

    iquam = xr.open_dataset(data_dir / 'IQUAM' / f'{year}{month:02d}-STAR-L2i_GHRSST-SST-iQuam-V2.10-v01.0-fv01.0.nc')

    quality = iquam.quality_level.values
    selection = quality >= 4

    id = iquam.platform_id.values[selection]
    type = iquam.platform_type.values[selection]
    lats = iquam.lat.values[selection]
    lons = iquam.lon.values[selection]
    values = iquam.sst.values[selection]

    months = iquam.month.values[selection].astype(int)
    days = iquam.day.values[selection].astype(int)

    dates = [datetime(2020, months[i], days[i]) for i in range(len(months))]

    climatology = xr.open_dataset(data_dir / "SST_CCI_climatology" / "SST_1x1_daily.nc")

    grid = gridder.Grid(2020, 10, id, lats, lons, dates, values, type, climatology)
    grid.make_5x5_grid_with_covariance()
    grid.plot_map()
    grid.plot_map5()
    grid.plot_map_unc5()


