import shutil
from typing import List, Tuple
import pandas as pd
import xarray as xr
import os
from datetime import timedelta
from dateutil.relativedelta import relativedelta

from src.configuration.config import Config
import cdsapi

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
        self.client.retrieve(dataset, request).download(f'download.grib')
        shutil.move("download.grib", Config.climate_grib_output_dir + variable_name + ".grib")

class GRIB_Reader:
    def __init__(self, filename, variable):
        # Load GRIB file
        self.ds = xr.open_dataset(filename, engine="cfgrib")

        self.variable_name = variable

        # print("Dataset:")
        # print(self.ds)
        # print("Variables:")
        # print(self.ds.variables)

    def read_timeseries(self, longitude: float, latitude: float, group_id: int):
        # logging.info(f"Processing (lon={longitude}, lat={latitude})")
        point_data = self.ds.sel({'latitude': latitude, 'longitude': longitude}, method='nearest').reset_index()

        if longitude < 0:
            longitude += 360

        ts = point_data[self.variable_name].to_dataframe()
        ts = ts.reset_index()

        logging.info(f"Timeseries for {self.variable_name}: {ts.columns}")

        ts.drop(['number', 'step', 'valid_time'], axis=1, inplace=True)

        ts['datetime'] = ts['time']
        ts['date'] = ts['datetime'].dt.date
        ts['time'] = ts['datetime'].dt.time
        ts.drop(['datetime'], axis=1, inplace=True)

        ts = ts.groupby('date')[self.variable_name].agg(['min', 'mean', 'max'])

        for column in ts.columns:
            ts.rename(columns={column: self.variable_name + '_' + column}, inplace=True)

        ts.reset_index(inplace=True)

        # logging.info("Reset index")
        # logging.info(ts)

        ts['date'] = pd.to_datetime(ts['date'])

        ts['group_id'] = group_id

        ts.set_index(['date', 'group_id'], inplace=True)

        return ts


class Downloader:
    def __init__(self):
        self.grib_downloader = GRIB_Downloader()

    def run(self, indexes: List[int], points: List[Tuple[float, float]]):
        dates = pd.date_range(start=Config.start_date, end=Config.end_date, freq="MS", inclusive="both")
        for long_name, short_name in Config.climate_data_param_names.items():
            for date in dates:
                start_date = date.strftime('%Y-%m-%d')
                end_date = (date + relativedelta(months=1) - timedelta(days=1)).strftime('%Y-%m-%d')

                logging.info(f"{long_name}: {date.year} {date.month}")

                self.grib_downloader.download_item(
                    variable_name=long_name, 
                    start_date=start_date,
                    end_date=end_date
                )

                grib_reader = GRIB_Reader(
                    filename=Config.climate_grib_output_dir + long_name + '.grib', 
                    variable=short_name
                )

                with Timer("Getting point data") as t:
                    ts = pd.concat([
                        grib_reader.read_timeseries(
                            longitude=lon,
                            latitude=lat,
                            group_id=idx
                        )
                        for idx, (lon, lat) in zip(indexes, points)
                    ], axis=1)

                ts.to_parquet(
                    path=Config.partitioned_climate_data_dir + long_name + "_" + str(date.year) + "_" + str(date.month) + '.parquet',
                    engine='pyarrow',
                    index=True
                )

                ### cleanup
                filenames_to_delete = os.listdir(Config.climate_grib_output_dir)

                for file_d in filenames_to_delete:
                    print(f"Deleting: {Config.climate_grib_output_dir + file_d}")
                    os.remove(Config.climate_grib_output_dir + file_d)

            # combine yearly data
            time_files = [file for file in os.listdir(Config.partitioned_climate_data_dir) if file[len(long_name)+1:][:4].isnumeric()]

            # parquet_writer = pq.ParquetWriter(where=Config.summary_dataset_filepath, schema=self.schema)

            combined_ts = pd.concat([
                pd.read_parquet(
                    path=Config.partitioned_climate_data_dir + time_file, 
                    engine="pyarrow"
                )
                for time_file in time_files
            ])



            for time_file in time_files:
                os.remove(path=Config.partitioned_climate_data_dir + time_file)

            combined_ts.to_parquet(
                path=Config.partitioned_climate_data_dir + long_name + '.parquet',
                engine='pyarrow',
                index=True
            )            




