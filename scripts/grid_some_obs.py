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
from itertools import product
import xarray as xr
import numpy as np
from pathlib import Path
import os
from datetime import datetime
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_dir = Path(os.getenv("DATADIR"))

    ts = []
    time = []

    hadsst4 = xr.open_dataset(data_dir / "ManagedData" / "Data" / "HadSST4" / "HadSST.4.1.1.0_median.nc")

    lat_slice = hadsst4.tos.sel(latitude=slice(-90, 90))
    lat_lon_slice = lat_slice.sel(longitude=slice(-180, 180))
    weights = np.cos(np.deg2rad(lat_lon_slice.latitude))
    weighted_mean = lat_lon_slice.weighted(weights).mean(("longitude", "latitude"))
    h4_time = np.arange(len(weighted_mean))/12. + 1850.

    climatology = xr.open_dataset(data_dir / "SST_CCI_climatology" / "SST_1x1_daily.nc")

    for year, month in product(range(1981, 2000), range(1, 13)):
        #year = 1989
        #month = 12
        print(year, month)

        file = data_dir / 'IQUAM' / f'{year}{month:02d}-STAR-L2i_GHRSST-SST-iQuam-V2.10-v01.0-fv01.0.nc'

        if not(file.exists()):
            continue

        iquam = xr.open_dataset(file)

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


        grid = gridder.Grid(2020, 10, id, lats, lons, dates, values, type, climatology)
        grid.make_5x5_grid_with_covariance()

        grid.plot_map(filename=data_dir/ "IQUAM" / "Figures" / f"one_deg_{year}{month:02d}.png")
        grid.plot_map5(filename=data_dir / "IQUAM" / "Figures" / f"five_deg_{year}{month:02d}.png")
        #grid.plot_covariance()
        grid.plot_map_unc5(filename=data_dir / "IQUAM" / "Figures" / f"unc_{year}{month:02d}.png")
        #grid.plot_covariance_row(-33.0, -142.0)

        gmsst = grid.calculate_area_average([-90, 90], [-180, 180])
        ts.append(gmsst)
        time.append(year+(month-1)/12.)
        print(gmsst)


    plt.plot(time, ts)
    plt.plot(h4_time, weighted_mean)
    plt.xlim(1975, 2010)
    plt.savefig(data_dir / "IQUAM" / "Figures" / "IQUAM_grid_average_time_series.png")
    plt.close()