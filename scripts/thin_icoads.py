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

import IMMA
from itertools import product
from pathlib import Path
import os

import pandas as pd

import warnings
warnings.simplefilter("ignore")

data_dir = Path(os.getenv("OODIR")) / "ICOADS"



for year, month in product(range(1850, 1985), range(1, 13)):
    print(year, month)

    filename = data_dir / f"IMMA1_R3.1.0_{year}-{month:02d}.gz"

    csv_filename = data_dir / f"icoads_{year}{month:02d}.csv"

    if csv_filename.exists():
        continue

    iobs = IMMA.get(str(filename))

    df = pd.DataFrame(
        {
            "uid": "ZZZ",
            "year": year,
            "month": month,
            "day": 1,
            "lat": 0.0,
            "lon": 0.0,
            "sst": 0.0 ,
            "id": "ZZZ",
            "hour": 0.0,
            "at": 0.0,
            "si": 0,
            "sid": 0,
            "dck": 0,
            "c1": "",
            "pt": 0,
            "sim": ""
        },
        index=[0]
    )
    count = 0

    for ob in iobs:

        row = []

        row.append(ob['UID'])
        row.append(ob['YR'])
        row.append(ob['MO'])
        row.append(ob['DY'])

        row.append(ob['LAT'])
        row.append(ob['LON'])
        row.append(ob['SST'])
        row.append(ob['ID'])

        row.append(ob['HR'])
        row.append(ob['AT'])
        row.append(ob['SI'])
        row.append(ob['SID'])
        row.append(ob['DCK'])
        row.append(ob['C1'])
        row.append(ob['PT'])

        try:
            sim = ob['SIM']
        except KeyError:
            sim = None
        row.append(sim)

        if None in row[0:8]:
            continue
        else:
            df.loc[count] = row
            count+=1

    df.to_csv(csv_filename)