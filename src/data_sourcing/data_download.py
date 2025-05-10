from typing import List
import pandas as pd
import xarray as xr


import logging
logging.basicConfig(level=logging.INFO)

# will maybe put later

class GRIB_Reader:
    def __init__(self, filename, variable_names):
        print(filename)
        print(variable_names)

        # Load GRIB file
        # self.ds = xr.open_dataset(filename, engine="cfgrib", backend_kwargs={"indexpath": ""})
        self.ds = xr.open_dataset(filename, engine="cfgrib", backend_kwargs={"filter_by_keys": {"shortName": variable_names}})

        self.variable_names = variable_names

        # Force eager loading of all variables
        # self.ds.load()

        # print("Dataset:")
        # print(self.ds)
        # print("Variables:")
        # print(self.ds.variables)

    def read_timeseries(self, longitude: float, latitude: float):
        if longitude < 0:
            longitude += 360

        total_df = pd.concat([
            self.convert_var_to_df(
                longitude=longitude,
                latitude=latitude,
                var_name=var_name
            )
            for var_name in self.variable_names
        ], axis=1)

        return total_df

    def convert_var_to_df(self, longitude: float, latitude: float, var_name: str):
        try:
            # Safely load just the variable first (no slicing)
            var = self.ds[var_name]

            # Trigger index creation fully
            var.load()

            # Now safely select the nearest point
            point_data = var.sel(latitude=latitude, longitude=longitude, method='nearest')

            ts = point_data.to_dataframe()
        except Exception as e:
            raise RuntimeError(f"Failed to convert variable '{var_name}' to DataFrame: {e}")
        
        ts = ts.reset_index()

        ts.drop(['number', 'step', 'valid_time'], axis=1, inplace=True)

        ts['datetime'] = ts['time']
        ts['date'] = ts['datetime'].dt.date
        ts['time'] = ts['datetime'].dt.time
        ts.drop(['datetime'], axis=1, inplace=True)

        # check if 'surface' is significant at all

        ts = ts.groupby('date')[var_name].agg(['min', 'mean', 'max']).reset_index()

        ts['date'] = pd.to_datetime(ts['date'])

        ts.set_index(['date'], inplace=True)

        for column in ts.columns:
            ts.rename(columns={column: var_name + '_' + column}, inplace=True)

        return ts

class Prober:
    def __init__(self, grib_filename: str):
        self.ds = xr.open_dataset(grib_filename, engine="cfgrib", backend_kwargs={"indexpath": ""})

        # logging.info("Dataset:")
        # logging.info(self.ds)
        logging.info("Variables:")
        logging.info(self.ds.variables)

from eccodes import codes_grib_new_from_file, codes_release, codes_get, codes_get_long
import sys

def check_grib_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            count = 0
            while True:
                gid = codes_grib_new_from_file(f)
                if gid is None:
                    break
                # Optional: check metadata
                param = codes_get(gid, 'shortName')
                level = codes_get_long(gid, 'level')
                logging.info(f"Message {count + 1}: {param} at level {level}")
                codes_release(gid)
                count += 1
        logging.info(f"File is valid with {count} GRIB message(s).")
        return True
    except Exception as e:
        logging.info(f"GRIB file check failed: {e}")
        return False

if __name__ == '__main__':
    # probe = Prober(grib_filename="data/dd8b880363f20653ba86802bdc6b4004.grib")

    grib_file, grib_vars = list(cams_files_and_vars.items())[3]

    if True or check_grib_file(grib_file):
        # import cfgrib
        # ds = cfgrib.open_dataset("data/86b429566a4d7af241a2afbf01efc0c.grib", indexpath="")
        # print(ds)
        # print(ds.variables)

        grib_reader = GRIB_Reader(filename=grib_file, variable_names=grib_vars)
        tsdf = grib_reader.read_timeseries(longitude=-74.006, latitude=40.7128)
        print(tsdf)
    else:
        print("Invalid!")