from datetime import datetime, timedelta
import time
from typing import Set, Tuple
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import os
from threading import Lock
import concurrent.futures
import logging
import os

from src.datamodels import PointGenerationModel, LandQueryModel
from src.utility import Timer
logging.basicConfig(level=logging.INFO)

from .services import NASAService
from .configuration.config import Config

class SummaryDataset:

    def __init__(self):

        self.climate_serv = NASAService()
        self.point_generator = PointGenerationModel()
        # self.google_earth_serv = GoogleEarthService()

        # 'landslide', 'flood', 'mass movement (dry)', 'extreme temperature ', 'storm', 'drought'

        self.schema = pa.schema([
            ("timestamp", pa.uint64()), # 'YYYYMMDD' to 0 - x
            ("group_id", pa.int64()),
            ("longitude", pa.float64()),
            ("latitude", pa.float64()),
            ("elevation", pa.float64()),
            ("has_landslide", pa.bool_()),
            ("has_flood", pa.bool_()),
            ("has_dry_mass_movement", pa.bool_()),
            ("has_extreme_temperature", pa.bool_()),
            ("has_storm", pa.bool_()),
            ("has_drought", pa.bool_()),
            ("num_deaths", pa.float64()),
            ("num_injured", pa.float64()),
            ("damage_cost", pa.float64()),
            ("avg_temperature_2m", pa.float64()),
            ("min_temperature_2m", pa.float64()),
            ("max_temperature_2m", pa.float64()),
            ("avg_dewfrostpoint_2m", pa.float64()), # from 'dewfrostpoint_2m'
            ("min_dewfrostpoint_2m", pa.float64()),
            ("max_dewfrostpoint_2m", pa.float64()),
            ("avg_precipitation", pa.float64()), # from 'precipitation'
            ("min_precipitation", pa.float64()),
            ("max_precipitation", pa.float64()),
            ("avg_windspeed_2m", pa.float64()), # only back to 1980/12/31
            ("min_windspeed_2m", pa.float64()),
            ("max_windspeed_2m", pa.float64()),
            ("avg_windspeed_10m", pa.float64()),
            ("min_windspeed_10m", pa.float64()),
            ("max_windspeed_10m", pa.float64()),
            ("avg_windspeed_50m", pa.float64()),
            ("min_windspeed_50m", pa.float64()),
            ("max_windspeed_50m", pa.float64()),
            ("avg_humidity_2m", pa.float64()), # from 'humidity_2m'
            ("min_humidity_2m", pa.float64()),
            ("max_humidity_2m", pa.float64()),
            ("avg_surface_pressure", pa.float64()), # from 'surface_pressure'
            ("min_surface_pressure", pa.float64()),
            ("max_surface_pressure", pa.float64()),
            ("avg_transpiration", pa.float64()), # from 'transpiration'
            ("min_transpiration", pa.float64()),
            ("max_transpiration", pa.float64()),
            ("avg_evaporation", pa.float64()), # from 'evaporation'
            ("min_evaporation", pa.float64()),
            ("max_evaporation", pa.float64()),
        ])

        self.create_table()

    def create_table(self):
        if os.path.exists(Config.summary_dataset_filepath):
            logging.info("Skipping Summary Parquet file creation.")
            return
        else:
            logging.info("Summary Parquet file does not exist. Creating now...")

        self.clear_dataset()

    def upload_data(self, start_date: datetime, end_date: datetime):
        # TODO: get more data by making the parameters more fine-grained
        points = self.point_generator.get_all_points()

        logging.info(f"Initial points len: {len(points)}")

        if Config.show_points:
            fig, ax = plt.subplots(figsize=(10, 8))
            LandQueryModel().gdf.geometry.plot(ax=ax, color='white', edgecolor='black', alpha=1.0, label="Dataset 1")
            list_points = np.array(list(points))
            plt.scatter(x=list_points[:, 0], y=list_points[:, 1], c='red')
            plt.show()

        points = self.fetch_nonexisting_list(points=set(points))

        logging.info(f"New points len: {len(points)}")

        if len(points) == 0:
            logging.info("No points to process summary data for! Skipping data uploading for summary dataset!")
            return

        dates = pd.date_range(start=start_date, end=end_date, freq="W", inclusive="both") \
            .strftime("%Y%m%d")

        logging.info(f"{len(points) * len(dates)} records to be added to the summary dataset")

        # population_df = self.google_earth_serv.get_population_data(points=points)

        thread_lock = Lock()

        # retrieve existing data so that we don't overwrite it
        existing_summary_data = pq.read_table(Config.summary_dataset_filepath)

        parquet_writer = pq.ParquetWriter(where=Config.summary_dataset_filepath, schema=self.schema)

        parquet_writer.write_table(table=existing_summary_data)

        num_tasks = len(points)

        def single_upload(longitude, latitude):
            try:
                logging.info(f"(long={longitude}, lat={latitude}): Processing...")
                with Timer("Single upload") as t:
                    climate_df = self.climate_serv.get_weekly_norm_data(longitude, latitude, start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"))

                    logging.info(f"(long={longitude}, lat={latitude}): Gathering disaster data...")
                    disaster_data = self.point_generator.get_data_for_point(longitude=longitude, latitude=latitude)

                    group_id = self.point_generator.get_group_id(longitude=longitude, latitude=latitude)

                    logging.info(f"(long={longitude}, lat={latitude}): Writing summary data to parquet file...")

                    batch_data = pa.table({
                        "timestamp": climate_df['timestamp'],
                        "group_id": [group_id] * len(dates),
                        "longitude": [longitude] * len(dates),
                        "latitude": [latitude] * len(dates),
                        "elevation": climate_df['elevation'],
                        "has_landslide": pa.array(disaster_data['has_landslide'], type=pa.bool_()),
                        "has_flood": pa.array(disaster_data['has_flood'], type=pa.bool_()),
                        "has_dry_mass_movement": pa.array(disaster_data['has_dry_mass_movement'], type=pa.bool_()),
                        "has_extreme_temperature": pa.array(disaster_data['has_extreme_temperature'], type=pa.bool_()),
                        "has_storm": pa.array(disaster_data['has_storm'], type=pa.bool_()),
                        "has_drought": pa.array(disaster_data['has_drought'], type=pa.bool_()),
                        "num_deaths": np.nan_to_num(x=disaster_data['total_deaths'].astype(dtype=np.float64), nan=0.0),
                        "num_injured": np.nan_to_num(x=disaster_data['num_injured'].astype(dtype=np.float64), nan=0.0),
                        "damage_cost": np.nan_to_num(x=disaster_data['damage_cost'].astype(dtype=np.float64), nan=0.0),
                        "avg_temperature_2m": climate_df["avg_temperature_2m"],
                        "min_temperature_2m": climate_df["min_temperature_2m"],
                        "max_temperature_2m": climate_df["max_temperature_2m"],
                        "avg_dewfrostpoint_2m": climate_df["avg_dewfrostpoint_2m"],
                        "min_dewfrostpoint_2m": climate_df["min_dewfrostpoint_2m"],
                        "max_dewfrostpoint_2m": climate_df["max_dewfrostpoint_2m"],
                        "avg_precipitation": climate_df["avg_precipitation"],
                        "min_precipitation": climate_df["min_precipitation"],
                        "max_precipitation": climate_df["max_precipitation"],
                        "avg_windspeed_2m": climate_df["avg_windspeed_2m"],
                        "min_windspeed_2m": climate_df["min_windspeed_2m"],
                        "max_windspeed_2m": climate_df["max_windspeed_2m"],
                        "avg_windspeed_10m": climate_df["avg_windspeed_10m"],
                        "min_windspeed_10m": climate_df["min_windspeed_10m"],
                        "max_windspeed_10m": climate_df["max_windspeed_10m"],
                        "avg_windspeed_50m": climate_df["avg_windspeed_50m"],
                        "min_windspeed_50m": climate_df["min_windspeed_50m"],
                        "max_windspeed_50m": climate_df["max_windspeed_50m"],
                        "avg_humidity_2m": climate_df["avg_humidity_2m"],
                        "min_humidity_2m": climate_df["min_humidity_2m"],
                        "max_humidity_2m": climate_df["max_humidity_2m"],
                        "avg_surface_pressure": climate_df["avg_surface_pressure"],
                        "min_surface_pressure": climate_df["min_surface_pressure"],
                        "max_surface_pressure": climate_df["max_surface_pressure"],
                        "avg_transpiration": climate_df["avg_transpiration"],
                        "min_transpiration": climate_df["min_transpiration"],
                        "max_transpiration": climate_df["max_transpiration"],
                        "avg_evaporation": climate_df["avg_evaporation"],
                        "min_evaporation": climate_df["min_evaporation"],
                        "max_evaporation": climate_df["max_evaporation"],
                    })
                    with thread_lock:
                        if not parquet_writer.is_open:
                            logging.info("Parquet writer is no longer open! Quitting...")
                            return
                        parquet_writer.write_table(batch_data)

                        logging.info(f"(long={longitude}, lat={latitude}): complete!")

                        nonlocal num_tasks
                        num_tasks -= 1
                        logging.info(f"Number of tasks remaining: {num_tasks}")
            except Exception as e:
                logging.error(e)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=Config.num_workers)

        for (longitude, latitude) in points:
            executor.submit(single_upload, longitude, latitude)

        while num_tasks > 0:
            try:
                time.sleep(15)
            except (KeyboardInterrupt, SystemExit):
                logging.info("Cancel initiated...")

                executor.shutdown(wait=True, cancel_futures=True)

                logging.info("Closing parquet file writer.")

                parquet_writer.close()

                logging.info("Stopped writing summary dataset.")

                return

        logging.info("Finished writing summary dataset!")

        if parquet_writer.is_open:
            parquet_writer.close()
    
    def fetch_nonexisting_list(self, points: Set):
        pq_table = pd.read_parquet(
            path=Config.summary_dataset_filepath, 
            engine='pyarrow',
            columns=['longitude', 'latitude']
        ).drop_duplicates(keep='first')

        existing_points = list(pq_table.itertuples(index=False, name=None))

        logging.info(f"Existing written points: {existing_points}")

        return list(points.difference(existing_points))
    
    def clear_dataset(self):
        # Create an empty table
        empty_table = pa.Table.from_batches([], schema=self.schema)

        # Write empty Parquet file
        pq.write_table(empty_table, Config.summary_dataset_filepath)

