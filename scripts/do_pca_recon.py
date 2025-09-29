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
from pathlib import Path
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import open_ocean.vbpca as vbpa
from open_ocean.utils import convert_climatology_to_ocean_areas

data_dir = Path(os.getenv('OODIR'))

data_file = data_dir / "IQUAM" / "oo_anomalies.nc"
data = xr.open_dataset(data_file)
recon = copy.deepcopy(data)

unc_file = data_dir / "IQUAM" / "oo_uncertainty.nc"
unc = xr.open_dataset(unc_file)

climatology = xr.open_dataset(data_dir / "SST_CCI_climatology" / "SST_1x1_daily.nc")
areas = convert_climatology_to_ocean_areas(climatology)

n_iterations = 159
n_eofs =  10
mask_percentage = 0.3333

interpolator = vbpa.VBPCA(
    data.sst.values,
    unc.sst.values,
    n_eofs,
    mask_percentage=mask_percentage,
    grid_areas=areas
)
interpolator.fit_model(max_iterations=n_iterations)

recon_data = interpolator.make_recon()
eofs = interpolator.make_eofs()
pc_series = interpolator.make_pc_series()

time = np.arange(data.sst.values.shape[0])/12. + 1981. + 8./12.

recon.sst.values[:] = recon_data[:]

plt.figure()
plt.gcf().set_size_inches(16, 9)
proj = ccrs.PlateCarree()
p = recon.sst[14].plot(
    transform=proj,
    subplot_kws={'projection': proj},
    levels=np.arange(-3, 3, 0.2),
    cmap='RdBu_r'
)
p.axes.coastlines()
plt.title("Reconstruction for November 1982")
plt.savefig(data_dir / "IQUAM" / "Figures" / "Reconstruction_November_1982.png")

for j in range(n_eofs):
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 9)

    axs[0].pcolormesh(eofs[j,:,:], cmap='RdBu_r')
    axs[0].set_title(f'EOF {j+1}')

    axs[1].plot(time, pc_series[j, :])
    axs[1].set_title(f'PC {j+1}')

    plt.savefig(data_dir / "IQUAM" / "Figures" / f"EOF{j}_PCs.png")