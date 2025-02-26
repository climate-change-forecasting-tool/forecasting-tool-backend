from datetime import datetime, timedelta
import time
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import numpy as np
import pandas as pd
import itertools
import os
from threading import Lock
import concurrent.futures
import logging
logging.basicConfig(level=logging.INFO)

from .services import DisasterDBService, NASAService, NOAAService


SUMMARY_DATASET_FILEPATH = 'db/summary_data.parquet'

"""
1. Read (& block) if a point has been written to the parquet file
2. 
"""

class SummaryDataset:

    def __init__(self):
        self.disaster_serv = DisasterDBService()
        self.climate_serv = NASAService()

        # Define a schema with column names and data types
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
        if os.path.exists(SUMMARY_DATASET_FILEPATH):
            logging.info("Skipping Parquet file creation.")
            return
        else:
            logging.info("Parquet file does not exist. Creating now...")

        self.clear_dataset()

    def upload_data(self, start_date: datetime, end_date: datetime):
        # points = self.generate_biased_points(long_side=20, lat_side=11) # TODO: more data
        points = [(-92.172935, 38.579201)]

        logging.info(f"Initial points len: {len(points)}")

        points = self.fetch_nonexisting_set(points=points)

        # logging.info(f"New points: {points}")
        logging.info(f"New points len: {len(points)}")

        dates = pd.date_range(start=start_date, end=end_date, freq="D", inclusive="both") \
            .strftime("%Y%m%d")

        logging.info(f"{len(points) * len(dates)} records to be added to the summary dataset")

        thread_lock = Lock()

        # parquet_writer = pq.ParquetWriter(where=SUMMARY_DATASET_FILEPATH, schema=self.schema)

        num_tasks = len(points)

        def single_upload(longitude, latitude):
            try:
                logging.info(f"(long={longitude}, lat={latitude}): Processing...")
                start_time = time.perf_counter()

                climate_df = NASAService.json_to_dataframe(
                    data=self.climate_serv.climate_query(longitude, latitude, dates[0], dates[-1]),
                    normalize_params=True
                )

                # Adjust timestamp to start from 0 and have an increment of 1 between contiguous times
                climate_df['timestamp'] = (pd.to_datetime(climate_df['timestamp'], format='%Y%m%d') - start_date).dt.days.astype(np.uint64)

                # logging.info(climate_df['timestamp'])
                # logging.info(climate_df['timestamp'].dtype)
                # logging.info(climate_df['timestamp'].astype(np.int64))
                # exit(0)

                logging.info(f"(long={longitude}, lat={latitude}): Gathering disaster data...")
                disaster_data = []
                for date in dates:
                    disaster_data.append(
                        self.disaster_serv.controller.query_spatiotemporal_point(longitude, latitude, date)[0] # TODO: maybe allow multiple
                    )
                disaster_data = np.array(disaster_data)
                logging.info(f"(long={longitude}, lat={latitude}): Writing summary data to parquet file...")
                batch_data = pa.table({
                    "timestamp": climate_df['timestamp'],
                    "longitude": [longitude] * len(dates), # climate_df['longitude']
                    "latitude": [latitude] * len(dates), # climate_df['latitude']
                    "elevation": climate_df['elevation'],
                    "disastertype": disaster_data[:, 0], 
                    "num_deaths": np.nan_to_num(x=disaster_data[:, 1].astype(dtype=np.float64), nan=0.0) / 100000000.0, # 100,000,000
                    "num_injuries": np.nan_to_num(x=disaster_data[:, 2].astype(dtype=np.float64), nan=0.0) / 100000000.0, # 100,000,000
                    "damage_cost": np.nan_to_num(x=disaster_data[:, 3].astype(dtype=np.float64), nan=0.0) / 1000000000000.0, # 1,000,000,000,000
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
                    # if not parquet_writer.is_open:
                    #     logging.info("Parquet writer is no longer open! Quitting...")
                    #     return
                    # parquet_writer.write_table(batch_data)
                    pq.write_to_dataset(table=batch_data, 
                                        root_path=SUMMARY_DATASET_FILEPATH,
                                        schema=self.schema,
                                        existing_data_behavior='delete_matching')

                    logging.info(f"(long={longitude}, lat={latitude}): complete!")

                    end_time = time.perf_counter()
                    total_time = int(end_time - start_time)
                    logging.info(f"Elapsed time: {int(total_time / 60):02d}:{total_time % 60:02d} mins")

                    nonlocal num_tasks
                    num_tasks -= 1
                    logging.info(f"Number of tasks remaining: {num_tasks}")
            except Exception as e:
                logging.error(e)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

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

                # logging.info("Closing parquet file writer.")

                # parquet_writer.close()

                logging.info("Stopped writing summary dataset.")

                return


        logging.info("Finished writing summary dataset!")

        # if parquet_writer.is_open:
        #     parquet_writer.close()

    def minmax_scaler(self, data, min_val, max_val):
        return (data - min_val) / (max_val - min_val)

    def generate_biased_points(self, long_side: int = 100, lat_side: int = 50):
        """
        Generates ``long_side`` * ``lat_side`` - ``long_side`` * 2 + 2 geographic points that are biased toward the equator around the world

        Args:
            long_side (int): number of vertical divisions across the world. 
                Includes the Prime Meridian if this is an even number.
            lat_side (int): number of horizontal divisions across the world. Ensure this is greater than 2.
                Best for this to be an odd number so that it has points on the equator.

        """
        
        # Generate longitude uniformly between -180 and 180, but exclude 180 from the bounds, b/c it is the same as -180
        longitudes = np.linspace(start=-180, stop=180, num=long_side, endpoint=False)
        
        # Generate latitude uniformly biased toward equator between -90 and 90
        tan_bound = np.pi / 4.
        latitudes = np.linspace(start=-tan_bound, stop=tan_bound, num=lat_side)

        latitudes = np.tan(latitudes) * 90.

        # logging.info(latitudes)

        points = list(itertools.product(
            longitudes, 

            # exclude poles from the cartesian product to exclude overlap
            latitudes[1:-1]
        ))

        # add poles
        points.append((0.0, -90.0))
        points.append((0.0, 90.0))
        
        return points
    
    def fetch_nonexisting_set(self, points):
        pq_table = pd.read_parquet(
            path=SUMMARY_DATASET_FILEPATH, 
            engine='pyarrow',
            columns=['longitude', 'latitude']
        ).drop_duplicates(keep='first')

        existing_points = list(pq_table.itertuples(index=False, name=None))

        logging.info(f"Existing written points: {existing_points}")

        return set(points).difference(existing_points)
    
    def clear_dataset(self):
        # Create an empty table
        empty_table = pa.Table.from_batches([], schema=self.schema)

        # Write empty Parquet file
        pq.write_table(empty_table, SUMMARY_DATASET_FILEPATH)

