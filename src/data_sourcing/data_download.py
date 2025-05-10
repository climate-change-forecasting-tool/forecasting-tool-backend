from typing import List
import pandas as pd
import xarray as xr

# will maybe put later







class Prober:
    def __init__(self, grib_filename: str):
        self.ds = xr.open_dataset(grib_filename, engine="cfgrib")

        print("Dataset:")
        print(self.ds)
        print("Variables:")
        print(self.ds.variables)

if __name__ == '__main__':
    probe = Prober(grib_filename="data/714aa50481381d97419b0527369f12ac.grib")

