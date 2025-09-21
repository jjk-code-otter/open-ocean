from pathlib import Path
import urllib.request
import os

import xarray as xa

data_dir = os.getenv("DATADIR")

for i in range(365):
    filename = f"D{i + 1:03d}-ESACCI-L4_GHRSST-SSTdepth-Climatology-GLOB_CDR3.0-v02.0-fv01.0.nc"

    url = (
        f"https://dap.ceda.ac.uk/neodc/eocis/data/"
        f"global_and_regional/sea_surface_temperature/CDR_v3/"
        f"Climatology/L4/v3.0.1/{filename}?download=1"
    )

    output_file = Path(data_dir) / "SST_CCI_climatology" / filename
    new_output_file = Path(data_dir) / "SST_CCI_climatology" / f"D{i + 1:03d}.nc"

    if not new_output_file.exists() and not output_file.exists():
        print(f"Downloading {filename}")
        urllib.request.urlretrieve(url, output_file)

        print(f"Processing {filename}")
        with xa.open_dataset(output_file) as dink:
            dink = dink.drop_vars(
                [
                    'analysed_sst_max',
                    'analysed_sst_min',
                    'sea_ice_fraction',
                    'sea_ice_fraction_max',
                    'sea_ice_fraction_min',
                    'sea_ice_fraction_std'
                ],
                errors='ignore'
            )

            print(f"Writing {new_output_file}")
            dink.to_netcdf(new_output_file)
            dink.close()

        output_file.unlink()

    print(filename)