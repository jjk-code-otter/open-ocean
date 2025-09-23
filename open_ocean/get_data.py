from itertools import product
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

from pathlib import Path
import os
import shutil

import cdsapi
import requests

def get_iquam_year_month(year, month):

    url = (f"https://star.nesdis.noaa.gov/pub/socd/sst/iquam/v2.10/"
           f"{year}{month:02d}-STAR-L2i_GHRSST-SST-iQuam-V2.10-v01.0-fv01.0.nc")

    data_dir = Path(os.getenv("DATADIR"))
    out_path = data_dir / 'IQUAM' / f'{year}{month:02d}-STAR-L2i_GHRSST-SST-iQuam-V2.10-v01.0-fv01.0.nc'

    if out_path.exists():
        print(f"File {out_path} already exists, skipping.")
        return

    try:
        r = requests.get(url, stream=True, headers={'User-agent': 'Mozilla/5.0'})

        if r.status_code == 200:
            with open(out_path, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)

    except requests.exceptions.ConnectionError:
        print(f"Couldn't connect to {url}")

    pass

def get_cds_year_month(year, month):

    dataset = "insitu-observations-surface-marine"
    request = {
        "variable": ["water_temperature"],
        "data_quality": ["passed"],
        "month": [
            f"{month:02d}"
        ],
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "year": [
            f"{year}"
        ]
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request).download()

if __name__ == '__main__':
    for year, month in product(range(1851, 2026), range(1, 13)):
        print(year, month)
        if year >= 1981:
            get_iquam_year_month(year, month)

        get_cds_year_month(year, month)