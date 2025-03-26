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
import math

from shapely import Point, Polygon

from src.utility import Timer
logging.basicConfig(level=logging.INFO)

from .services import DisasterDBService, NASAService, NOAAService, LandQueryService
from .configuration.config import Config

SUMMARY_DATASET_FILEPATH = 'db/summary_data.parquet'

"""
1. Read (& block) if a point has been written to the parquet file
2. 
"""

class SummaryDataset:

    def __init__(self, summary_dataset_filepath: str = None):
        self.summary_dataset_filepath = summary_dataset_filepath

        if summary_dataset_filepath is None:
            self.summary_dataset_filepath = SUMMARY_DATASET_FILEPATH

        self.disaster_serv = DisasterDBService()
        self.climate_serv = NASAService()
        self.land_query_serv = LandQueryService()

        # Define a schema with column names and data types
        # if os.path.exists(self.summary_dataset_filepath):
        #     self.schema = pq.read_table(self.summary_dataset_filepath).schema
        # else:
        self.schema = pa.schema([
            ("timestamp", pa.uint64()), # 'YYYYMMDD' to 0 - x
            ("longitude", pa.float64()),
            ("latitude", pa.float64()),
            ("elevation", pa.float64()),
            ("disastertype", pa.string()), 
            ("num_deaths", pa.float64()),
            ("num_injuries", pa.float64()),
            ("damage_cost", pa.float64()),
            ("avg_temperature_2m", pa.float64()),
            ("min_temperature_2m", pa.float64()),
            ("max_temperature_2m", pa.float64()),
            ("dewfrostpoint_2m", pa.float64()),
            ("precipitation", pa.float64()),
            ("avg_windspeed_2m", pa.float64()), # only back to 1980/12/31
            ("min_windspeed_2m", pa.float64()),
            ("max_windspeed_2m", pa.float64()),
            ("avg_windspeed_10m", pa.float64()),
            ("min_windspeed_10m", pa.float64()),
            ("max_windspeed_10m", pa.float64()),
            ("avg_windspeed_50m", pa.float64()),
            ("min_windspeed_50m", pa.float64()),
            ("max_windspeed_50m", pa.float64()),
            ("humidity_2m", pa.float64()),
            ("surface_pressure", pa.float64()),
            ("transpiration", pa.float64()),
            ("evaporation", pa.float64()),
        ])

        self.create_table()

    def create_table(self):
        if os.path.exists(self.summary_dataset_filepath):
            logging.info("Skipping Parquet file creation.")
            return
        else:
            logging.info("Parquet file does not exist. Creating now...")

        self.clear_dataset()

    def upload_data(self, start_date: datetime, end_date: datetime):
        # 2, 5, 0
        # TODO: get more data by making the parameters more fine-grained
        points = set(
            self.generate_cultured_points(
                inward_step=Config.pointgen_instep, 
                perimeter_step=Config.pointgen_circumstep, 
                absorption_radius=Config.pointgen_absorb_radius
            )
        )
        # points = [(-92.172935, 38.579201), (-89.172935, 38.579201), (-86.172935, 38.579201)]

        # logging.info(f"Initial points: {points}")
        logging.info(f"Initial points len: {len(points)}")

        if Config.show_points:
            fig, ax = plt.subplots(figsize=(10, 8))
            self.land_query_serv.gdf.geometry.plot(ax=ax, color='white', edgecolor='black', alpha=1.0, label="Dataset 1")
            list_points = np.array(list(points))
            plt.scatter(x=list_points[:, 0], y=list_points[:, 1], c='red')
            plt.show()

        points = self.fetch_nonexisting_set(points=points)

        # logging.info(f"New points: {points}")
        logging.info(f"New points len: {len(points)}")

        dates = pd.date_range(start=start_date, end=end_date, freq="D", inclusive="both") \
            .strftime("%Y%m%d")

        logging.info(f"{len(points) * len(dates)} records to be added to the summary dataset")

        thread_lock = Lock()

        # retrieve existing data so that we don't overwrite it
        existing_summary_data = pq.read_table(self.summary_dataset_filepath)

        parquet_writer = pq.ParquetWriter(where=self.summary_dataset_filepath, schema=self.schema)

        parquet_writer.write_table(table=existing_summary_data)

        num_tasks = len(points)

        def single_upload(longitude, latitude):
            try:
                logging.info(f"(long={longitude}, lat={latitude}): Processing...")
                with Timer("Single upload") as t:

                    climate_df = NASAService.json_to_dataframe( # takes like 4 seconds to do this
                        data=self.climate_serv.climate_query(longitude, latitude, dates[0], dates[-1]),
                        # normalize_params=True
                    )

                    # Adjust timestamp to start from 0 and have an increment of 1 between contiguous times
                    climate_df['timestamp'] = (pd.to_datetime(climate_df['timestamp'], format='%Y%m%d') - start_date).dt.days.astype(np.uint64)

                    logging.info(f"(long={longitude}, lat={latitude}): Gathering disaster data...")
                    disaster_data = []
                    for date in dates:
                        disaster_data.append(
                            self.disaster_serv.controller.query_spatiotemporal_point(longitude, latitude, date)[0] # TODO: maybe allow multiple
                        )
                    disaster_data = np.array(disaster_data)
                    logging.info(f"(long={longitude}, lat={latitude}): Writing summary data to parquet file...")
                    # TODO: Use Dask standard scaling instead of normalization; once we have all the data, then apply standardization afterward
                    # TODO: scale longitude and latitude with unit normalization; scale everything else with standard scaling
                    batch_data = pa.table({
                        "timestamp": climate_df['timestamp'],
                        "longitude": [longitude] * len(dates), # climate_df['longitude']
                        "latitude": [latitude] * len(dates), # climate_df['latitude']
                        "elevation": climate_df['elevation'],
                        "disastertype": disaster_data[:, 0], 
                        "num_deaths": np.nan_to_num(x=disaster_data[:, 1].astype(dtype=np.float64), nan=0.0), # 100,000,000
                        "num_injuries": np.nan_to_num(x=disaster_data[:, 2].astype(dtype=np.float64), nan=0.0), # 100,000,000
                        "damage_cost": np.nan_to_num(x=disaster_data[:, 3].astype(dtype=np.float64), nan=0.0), # 1,000,000,000,000
                        "avg_temperature_2m": climate_df['T2M'],
                        "min_temperature_2m": climate_df['T2M_MIN'],
                        "max_temperature_2m": climate_df['T2M_MAX'],
                        "dewfrostpoint_2m": climate_df['T2MDEW'],
                        "precipitation": climate_df['PRECTOTCORR'],
                        "avg_windspeed_2m": climate_df['WS2M'], # only back to 1980/12/31
                        "min_windspeed_2m": climate_df['WS2M_MIN'],
                        "max_windspeed_2m": climate_df['WS2M_MAX'],
                        "avg_windspeed_10m": climate_df['WS10M'],
                        "min_windspeed_10m": climate_df['WS10M_MIN'],
                        "max_windspeed_10m": climate_df['WS10M_MAX'],
                        "avg_windspeed_50m": climate_df['WS50M'],
                        "min_windspeed_50m": climate_df['WS50M_MIN'],
                        "max_windspeed_50m": climate_df['WS50M_MAX'],
                        "humidity_2m": climate_df['RH2M'],
                        "surface_pressure": climate_df['PS'],
                        "transpiration": climate_df['EVPTRNS'],
                        "evaporation": climate_df['EVLAND'],
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

        # 1 worker takes about 2:30-4:00 mins
        # 2 workers take about 5:00 mins to do one job each
        # 4 workers take about 8:30-10:00 mins to do one job each

        for (longitude, latitude) in points:
            executor.submit(single_upload, longitude, latitude)

        while num_tasks > 0:
            try:
                time.sleep(30)
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

    def minmax_scaler(self, data, min_val, max_val):
        return (data - min_val) / (max_val - min_val)
    
    def generate_cultured_points(self, inward_step: float, perimeter_step: float, absorption_radius: float, decimate_perc: float = 0.05):
        world_points = self.land_query_serv.get_world_points(inward_step=inward_step, perimeter_step=perimeter_step)

        # perhaps manipulate some points to fall within a region where a disaster from the training data happens

        # or merge points that are in a certain radius of eachother, then manipulate their locations slightly to fit in a known disaster location

        # TODO: need to combine points that are really close to eachother by creating a buffer and turning the resolution down

        # This polygon covers Northern Canada and Greenland; we remove '1-decimate_perc'*100% of points that lie in decimate polygon regions
        decimate_polygon = Polygon([(-10.4, 89.6), (-6.2, 84.0), (-21.8, 68.1), (-43.9, 59.0), (-65.2, 61.3), (-79.7, 63.6), (-94.4, 60.9), (-139.7, 60.5), (-141.1, 90.9)])
        frequency = math.ceil(1. / decimate_perc)
        # TODO: WE ACTUALLY DO NOT WANT ANY POINTS FROM NORTHERN CANADA OR GREENLAND
        def inbounds_decimate(point_coords: Tuple, idx: int):
            is_inside = decimate_polygon.contains(Point(point_coords))
            if is_inside:
                return idx % frequency == 0
            return True # keep all that are out of this bound

        world_points[:] = [point for idx, point in enumerate(world_points) if inbounds_decimate(point, idx)]

        return world_points
    
    def fetch_nonexisting_set(self, points: Set):
        pq_table = pd.read_parquet(
            path=self.summary_dataset_filepath, 
            engine='pyarrow',
            columns=['longitude', 'latitude']
        ).drop_duplicates(keep='first')

        existing_points = list(pq_table.itertuples(index=False, name=None))

        logging.info(f"Existing written points: {existing_points}")

        return points.difference(existing_points)
    
    def clear_dataset(self):
        # Create an empty table
        empty_table = pa.Table.from_batches([], schema=self.schema)

        # Write empty Parquet file
        pq.write_table(empty_table, self.summary_dataset_filepath)

