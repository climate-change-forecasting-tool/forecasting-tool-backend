from datetime import datetime
from typing import List
import pandas as pd
import xarray as xr

import logging
logging.basicConfig(level=logging.INFO)

class Prober:
    def __init__(self, grib_filename: str):
        self.ds = xr.open_dataset(grib_filename, engine="cfgrib", backend_kwargs={"indexpath": ""})

        # logging.info("Dataset:")
        # logging.info(self.ds)
        logging.info("Variables:")
        logging.info(self.ds.variables)




if __name__ == '__main__':
    # p = Prober(grib_filename="data/climate_data/total_column_ozone_2003_1.grib")

    file = pd.read_parquet(path="db/partitioned_data/total_column_water_vapour.parquet", engine="pyarrow")

    # logging.info(file['date'].max())

    file['date'] = pd.to_datetime(file['date'])
    # file['date'] = pd.to_datetime(file['date'], format="%Y/%m/%d")

    # file.groupby(['group_id', 'date'])

    logging.info(file[file['date'] > datetime(2024, 9, 30)])

    file.set_index(['date'], inplace=True)

    logging.info(file)

