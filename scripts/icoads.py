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

import os
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt

data_dir = Path(os.getenv("OODIR")) / "ICOADS"

ds = xr.open_dataset(data_dir / "ICOADS_R3.0.0_d185001_c20230502.nc")

plt.scatter(ds.SST, ds.lat)
plt.show()

print(ds)

print()