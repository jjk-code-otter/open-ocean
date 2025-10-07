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
import gzip

import xarray as xr
import pandas as pd
import numpy as np

import warnings
warnings.simplefilter("ignore")

data_dir = Path(os.getenv("OODIR")) / "ICOADS"

# Read the list of IDs to exclude. These are moorings which take large numbers of observations in some months
# but are not ships, drifting buoys or regular moored buoys. They are in lagoons, bayous, estuaries and similar
with open('list_of_ids_that_are_not_ships.txt', 'r') as f:
    forbidden_ids = [x.rstrip() for x in f.readlines()]

for year, month in product(range(2010, 2026), range(1, 13)):
    print(year, month)

    if year <= 2009:
        # Read and decipher the track IDs.
        track_file = data_dir / "KentTracks" / f"ICOADS_R3.0.0_{year}-{month:02d}_Tracks_Kent.nc"
        tracks = xr.open_dataset(track_file)
        track_ids = tracks.ID_Kent.values
        decoded_ids = []
        for i in range(track_ids.shape[1]):
            decoded_id = ''.join([(x.astype(str)) for x in track_ids[:, i]])
            decoded_ids.append(decoded_id.rstrip().lstrip())

    icoads_filename = data_dir / "IMMA1_R3.0.0" / f"IMMA1_R3.0.0_{year}-{month:02d}.gz"
    if year > 2014:
        icoads_filename = data_dir / "IMMA1_R3.0.0" / f"IMMA1_R3.0.2_{year}-{month:02d}.gz"
    csv_filename = data_dir / f"icoads_{year}{month:02d}.csv"

    if csv_filename.exists():
       continue

    iobs = IMMA.get(str(icoads_filename))

    data = {
        "uid": [], "year": [], "month": [], "day": [], "lat": [], "lon": [], "sst": [], "id": [], "hour": [], "at": [],
        "si": [], "sid": [], "dck": [], "c1": [], "pt": [], "sim": [], "snc": [], "trackid": [], "forbidden": [],
    }

    count = 0
    count_all = 0

    correspond = {
        'uid': 'UID', 'year': 'YR', 'month': 'MO', 'day': 'DY', 'lat': 'LAT', 'lon': 'LON', 'sst': 'SST', 'id': 'ID',
        'hour': 'HR', 'at': 'AT', 'si': 'SI', 'sid': 'SID', 'dck': 'DCK', 'c1': 'C1', 'pt': 'PT', 'snc': 'SNC',
    }

    for ob in iobs:
        for v, k in correspond.items():
            data[v].append(ob[k])

        if ob['ID'] is not None:
            id_stripped = ob['ID'].rstrip().lstrip()
            if id_stripped in forbidden_ids:
                data['forbidden'].append(1)
            else:
                data['forbidden'].append(0)
        else:
            data['forbidden'].append(0)

        try:
            sim = ob['SIM']
        except KeyError:
            sim = None

        data['sim'].append(sim)
        if year <= 2009:
            data['trackid'].append(decoded_ids[count_all])
        else:
            if ob['ID'] is not None:
                data['trackid'].append(id_stripped)
            else:
                data['trackid'].append('NA')

        count_all += 1

    df = pd.DataFrame(data)

    # Throw out anything we can't use
    df = df[~np.isnan(df['sst'])]
    df = df[~np.isnan(df['year'])]
    df = df[~np.isnan(df['month'])]
    df = df[~np.isnan(df['day'])]
    df = df[~np.isnan(df['lat'])]
    df = df[~np.isnan(df['lon'])]

    # Do not use weird moorings
    df = df[df['forbidden'] == 0]

    # No CMAN stations, no ice stations, no oceanographic, no etc.
    df = df[df['pt'] <= 7]

    # No oceanographic data or Deck 874 which is a mess
    df = df[df['dck'] != 874]
    df = df[df['dck'] != 780]

    df = df.drop(columns=['forbidden'])

    df.to_csv(csv_filename, float_format='%.2f')

    if year <= 2009:
        if count_all != len(decoded_ids):
            print("mismatch in length")
