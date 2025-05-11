import shutil
import time
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
from threading import Lock
import concurrent.futures
from threading import current_thread

from dotenv import load_dotenv

import logging

from src.utility import Timer
logging.basicConfig(level=logging.INFO)
# https://confluence.ecmwf.int/display/CKB/CAMS%3A+Reanalysis+data+documentation#CAMS:Reanalysisdatadocumentation-KnownissuesCAMSglobalreanalysis(EAC4)

class GRIB_Downloader:
    def __init__(self, account_number: int):
        load_dotenv('.env')

        key_name = 'ads_key' + str(account_number)

        logging.info(key_name)

        self.client = cdsapi.Client(
            url="https://ads.atmosphere.copernicus.eu/api",
            key=os.getenv(key_name)
        )

    def download_item(self, variable_name: str, start_date: str, end_date: str):
        try:
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
        except Exception as e:
            logging.error(e)
            raise e

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
        num_accounts = 2
        self.grib_downloaders = [GRIB_Downloader(account_number=i + 1) for i in range(num_accounts)]

        if Config.redownload_climate_data:
            for dir in [Config.climate_grib_output_dir, Config.partitioned_climate_data_dir]:
                files = os.listdir(path=dir)

                for file in files:
                    os.remove(dir + file)

        # # list duplicates
        # existing_files = dict()
        # for file in os.listdir(Config.climate_grib_output_dir):
        #     current_num = existing_files.get(file, 0)
        #     existing_files.update({file: current_num + 1})

        # for file, number in existing_files.items():
        #     if number > 1:
        #         logging.info(file)

        # logging.info(len(existing_files))
        # logging.info(sum(list(existing_files.values())))
        # exit(0)

        existing_var_files = [file[:-8] for file in os.listdir(Config.partitioned_climate_data_dir) if file[:-8] in list(Config.climate_data_param_names.keys())]
        # logging.info(f"Existing files: {existing_var_files}")
        self.remaining_vars = list(set(Config.climate_data_param_names.keys()).difference(existing_var_files))
        logging.info(f"Need to download: {self.remaining_vars}")

    def run(self, indexes: List[int], points: List[Tuple[float, float]]):
        dates = pd.date_range(start=Config.start_date, end=Config.end_date, freq="MS", inclusive="both")

        points_np = np.hstack([np.array(points), np.array(indexes)[:, np.newaxis]])

        points_np[np.where(points_np[:, 0] < 0), 0] += 360.

        points_df = pd.DataFrame(points_np, columns=["target_lon", "target_lat", "group_id"])

        files_to_download = [] # (long_name, short_name, start_date, end_date)

        for long_name in self.remaining_vars:
            for date in dates:
                start_date = date.strftime('%Y-%m-%d')
                end_date = (date + relativedelta(months=1) - timedelta(days=1)).strftime('%Y-%m-%d')

                filename = long_name + "_" + str(date.year) + "_" + str(date.month)

                if not os.path.exists(Config.partitioned_climate_data_dir + filename + ".parquet"):
                    if not os.path.exists(Config.climate_grib_output_dir + filename + ".grib"):
                        files_to_download.append((long_name, start_date, end_date))

        check_dlfiles_longnames = list(set([file_dl[0] for file_dl in files_to_download]))
        logging.info(f"Long names that still need to be downloaded: {check_dlfiles_longnames}")

        logging.info(f"Need to download: {files_to_download}")

        thread_lock = Lock()

        num_files_to_download = len(files_to_download)

        def download_file(long_name, start_date, end_date):
            try:
                thread_id = int(current_thread().getName()[-1])
                logging.info(f"Th{thread_id}: Processing {long_name} for {start_date}/{end_date}")
                self.grib_downloaders[thread_id % 2].download_item(
                    variable_name=long_name, 
                    start_date=start_date,
                    end_date=end_date
                )
                with thread_lock:
                    nonlocal num_files_to_download
                    num_files_to_download -= 1
                    logging.info(f"{num_files_to_download} remain that need to be downloaded.")
            except Exception as e:
                logging.error(e)
                raise e

        if len(files_to_download) > 0:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=Config.num_downloaders)

            for (long_name, start_date, end_date) in files_to_download:
                executor.submit(download_file, long_name, start_date, end_date)

            while num_files_to_download > 0:
                try:
                    time.sleep(15)
                except (KeyboardInterrupt, SystemExit):
                    logging.info("Cancel initiated...")

                    executor.shutdown(wait=True, cancel_futures=True)

                    logging.info("Stopped processing.")

                    return
            
            executor.shutdown(wait=True, cancel_futures=True)


        files_to_process = []
        for long_name in self.remaining_vars:
            for date in dates:
                start_date = date.strftime('%Y-%m-%d')
                end_date = (date + relativedelta(months=1) - timedelta(days=1)).strftime('%Y-%m-%d')

                filename = long_name + "_" + str(date.year) + "_" + str(date.month)

                if not os.path.exists(Config.partitioned_climate_data_dir + filename + ".parquet"):
                    files_to_process.append((filename, long_name))
        # logging.info(f"Need to process: {files_to_process}")

        for (filename, long_name) in files_to_process:
            if not os.path.exists(Config.climate_grib_output_dir + filename + ".grib"):
                logging.info(f"Skipping {filename}")
                continue
            logging.info(f"Processing {filename}")

            grib_reader = GRIB_Reader(
                filename=Config.climate_grib_output_dir + filename + '.grib', 
                variable=Config.climate_data_param_names.get(long_name)
            )

            ts = grib_reader.read_timeseries(
                points_df=points_df
            )

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
        
        # for long_name in list(Config.climate_data_param_names.keys()):
        #     logging.info(f"Combining {long_name}")

        #     # combine yearly data
        #     for file in os.listdir(Config.partitioned_climate_data_dir):
        #         if file[:len(long_name)] == long_name:
        #             print(file[len(long_name)+1:][:4])

        #     time_files = [file for file in os.listdir(Config.partitioned_climate_data_dir) if file[:len(long_name)] == long_name and file[len(long_name)+1:][:4].isnumeric()]

        #     combined_ts = pd.concat([
        #         pd.read_parquet(
        #             path=Config.partitioned_climate_data_dir + time_file, 
        #             engine="pyarrow"
        #         )
        #         for time_file in time_files
        #     ])

        #     combined_ts.sort_values(by=['date'])

        #     logging.info("Combined time data")
        #     logging.info(combined_ts)

        #     # for time_file in time_files:
        #     #     os.remove(path=Config.partitioned_climate_data_dir + time_file)

        #     combined_ts.to_parquet(
        #         path=Config.partitioned_climate_data_dir + long_name + '.parquet',
        #         engine='pyarrow',
        #         index=True
        #     )

