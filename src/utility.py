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
    def __enter__(self):
        self.start = perf_counter()
        return self
    
    def __exit__(self, type, value, traceback):
        self.elapsed_time = perf_counter() - self.start
        self.output = f"Elapsed time: {int(self.elapsed_time / 60):02d}:{self.elapsed_time % 60:02d} mins"
        logging.info(self.output)