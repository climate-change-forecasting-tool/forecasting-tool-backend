from datetime import datetime
import itertools
from typing import Any, List
import pandas as pd
from functools import reduce
from time import perf_counter
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)

def flatten_list(l: List[Any]) -> List[Any]:
    copy = []

    for item in l:
        if isinstance(item, List):
            copy.extend(item)
        else:
            copy.append(item)
    return list(copy)

class Timer:
    def __init__(self, name: str = None):
        self.name = name

    def __enter__(self):
        self.start = perf_counter()
        return self
    
    def __exit__(self, type, value, traceback):
        self.elapsed_time = perf_counter() - self.start
        self.output = f"{'('+self.name+') ' if self.name else ''}Elapsed time: {int(self.elapsed_time / 60):02d}:{int(self.elapsed_time % 60):02d}"
        logging.info(self.output)

def get_nonunique_elements(arr):
    """
    Returns nonunique elements of a NumPy array.

    Args:
        arr (np.ndarray): Input NumPy array.

    Returns:
        np.ndarray: Array containing nonunique elements.
    """
    unique_elements, counts = np.unique(arr, return_counts=True)
    nonunique_elems = arr[np.isin(arr, unique_elements[counts > 1])]
    return np.unique(nonunique_elems)

def latlon_to_xyz(lat, lon):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return x, y, z


def join_dataframes_on_index(dataframes: List[pd.DataFrame]):
    """
    Joins a list of Pandas DataFrames on their indexes using reduce.

    Args:
        dataframes: A list of Pandas DataFrames.

    Returns:
        A single Pandas DataFrame resulting from the join operation, 
        or None if the input list is empty.
    """
    if not dataframes:
        return None

    # Use reduce to successively join DataFrames on their indexes
    joined_df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), dataframes)
    return joined_df

def get_astronomical_season(date: datetime, latitude: float):
    year = date.year
    
    # Astronomical season boundaries (approximate, adjust for leap years if needed)
    spring_start = datetime(year, 3, 20)
    summer_start = datetime(year, 6, 21)
    fall_start = datetime(year, 9, 22)
    winter_start = datetime(year, 12, 21)

    seasons = ['spring', 'summer', 'fall', 'winter']
    hemisphere_offset = 0 if (latitude >= 0) else 2

    season_idx = 3
    if spring_start <= date < summer_start:
        season_idx = 0 # spring (northern)
    elif summer_start <= date < fall_start:
        season_idx = 1 # summer (northern)
    elif fall_start <= date < winter_start:
        season_idx = 2 # fall (northern)
    else:
        season_idx = 3 # winter (northern)

    return seasons[(season_idx + hemisphere_offset) % 4]

def get_astronomical_season_df(row):
    return get_astronomical_season(
        date=pd.to_datetime(row["timestamp"]), 
        latitude=row["latitude"]
    )