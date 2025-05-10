from typing import List
import pandas as pd
import xarray as xr


import logging
logging.basicConfig(level=logging.INFO)

# will maybe put later

class GRIB_Reader:
    def __init__(self, filename, variable_name):
        # Load GRIB file
        self.ds = xr.open_dataset(filename, engine="cfgrib")

        self.variable_name = variable_name

        # print("Dataset:")
        # print(self.ds)
        # print("Variables:")
        # print(self.ds.variables)

    def convert_var_to_df(self, longitude: float, latitude: float):
        if longitude < 0:
            longitude += 360

        try:
            # Safely load just the variable first (no slicing)
            var = self.ds[self.variable_name]

            # Now safely select the nearest point
            point_data = var.sel(latitude=latitude, longitude=longitude, method='nearest')

            ts = point_data.to_dataframe()
        except Exception as e:
            raise RuntimeError(f"Failed to convert variable '{self.variable_name}' to DataFrame: {e}")
        
        ts = ts.reset_index()

        ts.drop(['number', 'step', 'valid_time'], axis=1, inplace=True)

        ts['datetime'] = ts['time']
        ts['date'] = ts['datetime'].dt.date
        ts['time'] = ts['datetime'].dt.time
        ts.drop(['datetime'], axis=1, inplace=True)

        # check if 'surface' is significant at all

        ts = ts.groupby('date')[self.variable_name].agg(['min', 'mean', 'max']).reset_index()

        ts['date'] = pd.to_datetime(ts['date'])

        ts.set_index(['date'], inplace=True)

        for column in ts.columns:
            ts.rename(columns={column: self.variable_name + '_' + column}, inplace=True)

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
    
from eccodes import codes_grib_new_from_file, codes_get, codes_write, codes_release
import os

def split_grib_file(input_file):
    output_dir = 'data/split_clim_data'
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'rb') as f:
        while True:
            gid = codes_grib_new_from_file(f)
            if gid is None:
                break

            # Use shortName (e.g., t2m, u10, v10) to name files
            short_name = codes_get(gid, 'shortName')
            output_path = os.path.join(output_dir, f'split_{short_name}.grib')

            # Append to the output file
            with open(output_path, 'ab') as out:
                codes_write(gid, out)

            codes_release(gid)

if __name__ == '__main__':
    # split_grib_file(input_file = 'data/climate_data/2m_temperature.grib')

    gr = GRIB_Reader(filename="data/split_clim_data/split_2t.grib", variable_name='t2m')

    ts = gr.convert_var_to_df(
        longitude=-74.006,
        latitude=40.7128
    )

    print(ts)

    # probe = Prober(grib_filename="data/climate_data/becd83d22c97f363d8a9860b806b37b.grib")


    # grib_file, grib_vars = list(cams_files_and_vars.items())[3]

    # if True or check_grib_file(grib_file):
    #     # import cfgrib
    #     # ds = cfgrib.open_dataset("data/86b429566a4d7af241a2afbf01efc0c.grib", indexpath="")
    #     # print(ds)
    #     # print(ds.variables)

    #     grib_reader = GRIB_Reader(filename=grib_file, variable_names=grib_vars)
    #     tsdf = grib_reader.read_timeseries(longitude=-74.006, latitude=40.7128)
    #     print(tsdf)
    # else:
    #     print("Invalid!")

    pass