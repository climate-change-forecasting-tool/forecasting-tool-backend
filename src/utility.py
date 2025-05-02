import itertools
from typing import Any, List

def multiget(d: dict, levels: List[str], default = None) -> Any:
    """
    Retrieves a specified value from a nested dictionary
    """
    dummy = d
    for level in levels[:-1]:
        dummy = dummy.get(level, {})
        if not dummy:
            return default
    return dummy.get(levels[-1], default)

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