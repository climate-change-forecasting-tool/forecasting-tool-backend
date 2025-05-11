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
    # p = Prober(grib_filename="")

    file = pd.read_parquet(path="db/partitioned_data/2m_temperature.parquet", engine="pyarrow")

    file.groupby(['group_id', 'date'])

    logging.info(file)