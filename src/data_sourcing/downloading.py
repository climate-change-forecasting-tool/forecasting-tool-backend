import shutil
from typing import List, Tuple
import pandas as pd
import xarray as xr
import os
from datetime import timedelta, datetime
from dateutil.relativedelta import relativedelta
import numpy as np
from src.configuration.config import Config
import cdsapi
from scipy.spatial import cKDTree

from dotenv import load_dotenv

import logging

from src.utility import Timer
logging.basicConfig(level=logging.INFO)
# https://confluence.ecmwf.int/display/CKB/CAMS%3A+Reanalysis+data+documentation#CAMS:Reanalysisdatadocumentation-KnownissuesCAMSglobalreanalysis(EAC4)

class GRIB_Downloader:
    def __init__(self):
        load_dotenv('.env')

        self.client = cdsapi.Client(
            url="https://ads.atmosphere.copernicus.eu/api",
            key=os.getenv('ads_key')
        )

    def download_item(self, variable_name: str, start_date: str, end_date: str):
        dataset = "cams-global-reanalysis-eac4"
        request = {
            "variable": [
                variable_name
            ],
            "date": [f"{start_date}/{end_date}"],
            "time": [
                "00:00", "06:00", "12:00", "18:00"
            ],
            "data_format": "grib"
        }
        date_f = datetime.strptime(start_date, '%Y-%m-%d')
        download_name = variable_name + "_" + str(date_f.year) + "_" + str(date_f.month) + ".grib"
        self.client.retrieve(dataset, request).download(download_name)
        shutil.move(download_name, Config.climate_grib_output_dir + download_name)

class GRIB_Reader:
    def __init__(self, filename, variable):
        # Load GRIB file
        self.ds = xr.open_dataset(filename, engine="cfgrib")
        self.variable_name = variable

    def read_timeseries(self, points_df: pd.DataFrame):
        df = self.ds[self.variable_name].to_dataframe().reset_index()

        df.drop(['number', 'step', 'surface', 'valid_time'], axis=1, inplace=True)

        grid_df = df[["latitude", "longitude"]].drop_duplicates().reset_index()

        tree = cKDTree(grid_df[["latitude", "longitude"]].values)
        distances, indices = tree.query(points_df[["target_lat", "target_lon"]].values)

        # Add nearest lat/lon columns to points_df
        nearest_points = grid_df.loc[indices].reset_index(drop=True)
        points_df = pd.concat([points_df, nearest_points.rename(columns={"latitude": "lat", "longitude": "lon"})], axis=1)

        final_df = df.merge(
            points_df[["group_id", "lat", "lon"]],
            left_on=["latitude", "longitude"],
            right_on=["lat", "lon"],
            how="inner"
        )

        final_df['date'] = pd.to_datetime(final_df['time']).dt.date
        agg = (
            final_df.groupby(['date', 'group_id'])[self.variable_name]
            .agg(['min', 'mean', 'max'])
            .rename(columns=lambda c: f"{self.variable_name}_{c}")
            .reset_index()
        )

        logging.info(f"agg: {agg[['group_id', 'date']].groupby(['group_id']).count().agg(['max', 'min'])}")

        return agg


class Downloader:
    def __init__(self):
        self.grib_downloader = GRIB_Downloader()

        if Config.redownload_climate_data:
            for dir in [Config.climate_grib_output_dir, Config.partitioned_climate_data_dir]:
                files = os.listdir(path=dir)

                for file in files:
                    os.remove(dir + file)

        existing_var_files = [file[:-8] for file in os.listdir(Config.partitioned_climate_data_dir)]
        logging.info(f"Existing files: {existing_var_files}")
        self.remaining_vars = list(set(Config.climate_data_param_names.keys()).difference(existing_var_files))
        logging.info(f"Need to download: {self.remaining_vars}")

    def run(self, indexes: List[int], points: List[Tuple[float, float]]):
        dates = pd.date_range(start=Config.start_date, end=Config.end_date, freq="MS", inclusive="both")

        points_np = np.hstack([np.array(points), np.array(indexes)[:, np.newaxis]])

        points_np[np.where(points_np[:, 0] < 0), 0] += 360.

        points_df = pd.DataFrame(points_np, columns=["target_lon", "target_lat", "group_id"])

        for long_name in self.remaining_vars:
            logging.info(f"Starting download process for {long_name}")
            with Timer(f"Date cycle for {long_name}"):
                for date in dates:
                    start_date = date.strftime('%Y-%m-%d')
                    end_date = (date + relativedelta(months=1) - timedelta(days=1)).strftime('%Y-%m-%d')

                    filename = long_name + "_" + str(date.year) + "_" + str(date.month)

                    logging.info(f"{long_name}: {date.year} {date.month}")

                    if os.path.exists(Config.partitioned_climate_data_dir + filename + ".parquet"):
                        logging.info("Skipping because it already exists...")
                        continue

                    self.grib_downloader.download_item(
                        variable_name=long_name, 
                        start_date=start_date,
                        end_date=end_date
                    )

                    grib_reader = GRIB_Reader(
                        filename=Config.climate_grib_output_dir + filename + '.grib', 
                        variable=Config.climate_data_param_names.get(long_name)
                    )

                    with Timer(f"Getting point data ({long_name}; {date})") as t:
                        ts = grib_reader.read_timeseries(
                            points_df=points_df
                        )

                    logging.info("Output")
                    logging.info(ts)

                    ts.to_parquet(
                        path=Config.partitioned_climate_data_dir + filename + '.parquet',
                        engine='pyarrow',
                        index=True
                    )

                    ### cleanup
                    filenames_to_delete = [file for file in os.listdir(Config.climate_grib_output_dir) if file[:len(filename)] == filename]

                    for file_d in filenames_to_delete:
                        logging.info(f"Deleting: {Config.climate_grib_output_dir + file_d}")
                        os.remove(Config.climate_grib_output_dir + file_d)

                # combine yearly data
                time_files = [file for file in os.listdir(Config.partitioned_climate_data_dir) if file[:len(long_name)] == long_name and file[len(long_name)+1:][:4].isnumeric()]

                # parquet_writer = pq.ParquetWriter(where=Config.summary_dataset_filepath, schema=self.schema)

                combined_ts = pd.concat([
                    pd.read_parquet(
                        path=Config.partitioned_climate_data_dir + time_file, 
                        engine="pyarrow"
                    )
                    for time_file in time_files
                ])

                logging.info("Combined time data")
                logging.info(combined_ts)

                for time_file in time_files:
                    os.remove(path=Config.partitioned_climate_data_dir + time_file)

                combined_ts.to_parquet(
                    path=Config.partitioned_climate_data_dir + long_name + '.parquet',
                    engine='pyarrow',
                    index=True
                )            




