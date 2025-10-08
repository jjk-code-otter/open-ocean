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
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt

data_dir = Path(os.getenv("OODIR"))

count_dict = {}

for year in range(2010, 2026):
    print(year)
    for month in range(1, 13):
        if (data_dir / "ICOADS" / f"icoads_{year}{month:02d}.csv").exists():
            df = pd.read_csv(data_dir / "ICOADS" / f"icoads_{year}{month:02d}.csv")

            # Select buoys that
            selection =(df['sst'] < -2.0)  & (np.isin(df['pt'], [6, 7]))

            cold = df[selection]
            cold_groups = cold.groupby('id').size()

            for a, b in cold_groups.items():
                if a not in count_dict:
                    count_dict[a] = b
                else:
                    count_dict[a] += b

with open('arctic_buoy_list_2.txt', 'w') as f:
    for k, v in count_dict.items():
        if v > 5:
            f.write(k+'\n')