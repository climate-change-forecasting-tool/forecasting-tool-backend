from datetime import datetime, timedelta
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import itertools
import os
from threading import Lock
import concurrent.futures
import logging
logging.basicConfig(level=logging.INFO)

from .services import DisasterDBService, NASAService, NOAAService


SUMMARY_DATASET_FILEPATH = 'data/summary_data.parquet'
SUMMARY_DATASET_IDX_FILEPATH = 'src/dataset_summary_saveidx.txt'

# parallelize this


"""
1. Read (& block) if a point has been written to the parquet file
2. 
"""

class SummaryDataset:
    log_rate = 100

    def __init__(self):
        self.disaster_serv = DisasterDBService()
        self.climate_serv = NASAService()

        # Define a schema with column names and data types
        self.schema = pa.schema([
            ("timestamp", pa.string()), # 'YYYYMMDD'
            ("longitude", pa.float64()),
            ("latitude", pa.float64()),
            ("elevation", pa.float64()),
            ("disastertype", pa.string()), 
            ("num_deaths", pa.int64()),
            ("num_injuries", pa.int64()),
            ("damage_cost", pa.float64()),
            ("avg_temperature_2m", pa.float32()),
            ("min_temperature_2m", pa.float32()),
            ("max_temperature_2m", pa.float32()),
            ("dewfrostpoint_2m", pa.float32()),
            ("precipitation", pa.float32()),
            ("avg_windspeed_2m", pa.float32()), # only back to 1980/12/31
            ("min_windspeed_2m", pa.float32()),
            ("max_windspeed_2m", pa.float32()),
            ("avg_windspeed_10m", pa.float32()),
            ("min_windspeed_10m", pa.float32()),
            ("max_windspeed_10m", pa.float32()),
            ("avg_windspeed_50m", pa.float32()),
            ("min_windspeed_50m", pa.float32()),
            ("max_windspeed_50m", pa.float32()),
            ("humidity_2m", pa.float32()),
            ("surface_pressure", pa.float32()),
            ("transpiration", pa.float32()),
            ("evaporation", pa.float32()),
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
        points = self.generate_biased_points(long_side=20, lat_side=11) # TODO: more data

        parquet_writer = pq.ParquetWriter(SUMMARY_DATASET_FILEPATH, self.schema)

        dates = pd.date_range(start=start_date, end=end_date, freq="D", inclusive="both") \
            .strftime("%Y%m%d")
        
        choose_idx = 0
        with open(SUMMARY_DATASET_IDX_FILEPATH, "r") as file:
            choose_idx = int(file.readline()) + 1 # set the number in the file to -1 initially

        for idx, (longitude, latitude) in enumerate(points):
            if idx < choose_idx:
                continue

            logging.info(f"Processing (longitude={longitude}, latitude={latitude})")

            climate_df = NASAService.json_to_dataframe(
                self.climate_serv.climate_query(longitude, latitude, dates[0], dates[-1])
            )
            logging.info(f"Gathering disaster data...")
            disaster_data = []
            for date in dates:
                disaster_data.append(
                    self.disaster_serv.controller.query_spatiotemporal_point(longitude, latitude, date)[0] # TODO: maybe allow multiple
                )
            disaster_data = np.array(disaster_data)
            logging.info(f"Writing summary data to parquet file...")
            batch_data = pa.table({
                "timestamp": dates,
                "longitude": climate_df['longitude'],
                "latitude": climate_df['latitude'],
                "elevation": climate_df['elevation'],
                "disastertype": disaster_data[:, 0], 
                "num_deaths": np.nan_to_num(x=disaster_data[:, 1].astype(dtype=np.float64), nan=0.0).astype(dtype=np.int64),
                "num_injuries": np.nan_to_num(x=disaster_data[:, 2].astype(dtype=np.float64), nan=0.0).astype(dtype=np.int64),
                "damage_cost": np.nan_to_num(x=disaster_data[:, 3].astype(dtype=np.float64), nan=0.0),
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
            parquet_writer.write_table(batch_data)
            logging.info(f"Point ({longitude}, {latitude}) complete!")

            with open(SUMMARY_DATASET_IDX_FILEPATH, "w") as file:
                file.write(f"{idx}")

        logging.info("Finished writing summary dataset!")
        parquet_writer.close()

    def generate_biased_points(self, long_side: int = 100, lat_side: int = 50):
        """
        Generates ``long_side`` * ``lat_side`` geographic points that are biased toward the equator around the world

        Args:
            long_side (int): number of vertical divisions across the world. 
                Includes the Prime Meridian if this is an even number.
            lat_side (int): number of horizontal divisions across the world. Ensure this is greater than 2.
                Best for this to be an odd number so that it has points on the equator.

        """
        
        # Generate longitude uniformly between -180 and 180, but exclude 180 from the bounds, b/c it is the same as -180
        longitudes = np.linspace(start=-180, stop=180, num=long_side, endpoint=False)

        # logging.info(longitudes)
        
        # Generate latitude uniformly biased toward equator between -90 and 90
        tan_bound = np.pi / 4.
        latitudes = np.linspace(start=-tan_bound, stop=tan_bound, num=lat_side)

        latitudes = np.tan(latitudes) * 90.

        # logging.info(latitudes)

        # exclude poles from the cartesian product to exclude overlap
        points = list(itertools.product(longitudes, latitudes[1:-1]))

        # add poles
        points.append((0, -90.0))
        points.append((0, 90.0))
        
        return points
    
    def clear_dataset(self):
        # Create an empty table
        empty_table = pa.Table.from_batches([], schema=self.schema)

        # Write empty Parquet file
        pq.write_table(empty_table, SUMMARY_DATASET_FILEPATH)

