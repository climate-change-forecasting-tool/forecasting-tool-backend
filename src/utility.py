import itertools
from typing import Any, List

import pandas as pd

def flatten_list(l: List[Any]) -> List[Any]:
    copy = []

    for item in l:
        if isinstance(item, List):
            copy.extend(item)
        else:
            copy.append(item)
    return list(copy)

from time import perf_counter
import logging

logging.basicConfig(level=logging.INFO)

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

import numpy as np

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


from functools import reduce
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