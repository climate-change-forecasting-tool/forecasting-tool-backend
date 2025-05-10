import os
from typing import List, Tuple
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from src.utility import Timer

from src.configuration.config import Config
from .climate_model import GRIB_Controller

import logging
logging.basicConfig(level=logging.INFO)


"""
process:
1. Generate carpet of polygons (h3 hexagons)
2. Load all disaster geometries
3. Intersect carpet of polygons with ALL disaster geometries
    a. Remove polygon from carpet of polygons if it doesn't intersect (?)
4. Generate random point inside each polygon
5. Create new database with points and disaster data
"""

class SummaryDataset:
    # We will save the data produced by this and load it afterward; will just be a db, not gdb
    def __init__(self): # lower resolution = larger area

        self.climate_params = []

        for climate_var_name in Config.climate_data_param_names.keys():
            self.climate_params.append((climate_var_name + '_min', pa.float64()))
            self.climate_params.append((climate_var_name + '_mean', pa.float64()))
            self.climate_params.append((climate_var_name + '_max', pa.float64()))


        self.schema = pa.schema([
            ("timestamp", pa.uint64()),
            ("group_id", pa.int64()),
            ("longitude", pa.float64()),
            ("latitude", pa.float64()),
        ] + self.climate_params)

    def generate(self):
        if os.path.exists(Config.summary_dataset_filepath):
            if Config.recreate_summary_dataset:
                logging.info("Clearing Summary Dataset Parquet file and recreating now...")
                self.clear_dataset()
            else:
                logging.info("Skipping Summary Dataset Parquet file creation.")
                return
        else:
            logging.info("Summary Dataset Parquet file does not exist. Creating now...")
            self.clear_dataset()

        partitioned_files = [f for f in os.listdir(Config.partitioned_climate_data_dir)]
        file_variables = [v[:-5] for v in partitioned_files]

        total_df = pd.concat([
            pd.read_parquet(
                path=Config.partitioned_climate_data_dir + partitioned_file, 
                engine="pyarrow"
            )
            for partitioned_file in partitioned_files
        ], axis=1)

        # parquet_writer = pq.ParquetWriter(where=Config.summary_dataset_filepath, schema=self.schema)

        print(total_df)

        total_df.to_parquet(path=Config.summary_dataset_filepath, engine='pyarrow', index=True)

        # # saving longitude, latitude, dates, disastertype, num deaths, num injured, property damage cost
        # logging.info("Saving values...")
        # with Timer("Saving values"):
        #     for idx, (h3_index, (lon, lat)) in enumerate(zip(h3_int_indexes, h3_centroids)):
        #         with Timer(f"lon={lon}, lat: {lat}; group_id: {h3_index}; ({idx+1}/{len(h3_int_indexes)})"):
        #             climate_df = grib_controller.get_point_data(
        #                 longitude=lon,
        #                 latitude=lat
        #             )

        #             # print("Climate df:")
        #             # print(climate_df)

        #             # print(climate_df.index.to_series())

        #             dates = (climate_df.index.to_series() - Config.start_date).dt.days.astype(int)

        #             # print(dates)

        #             entry_dict = dict({
        #                 "timestamp": pa.array(dates, type=pa.uint64()),
        #                 "group_id": pa.array([h3_index] * len(dates), type=pa.int64()),
        #                 "longitude": pa.array([lon] * len(dates), type=pa.float64()),
        #                 "latitude": pa.array([lon] * len(dates), type=pa.float64()),
        #             })

        #             for climate_param, pa_type in self.climate_params:
        #                 entry_dict.update({climate_param: pa.array(climate_df[climate_param], type=pa_type)})

        #             entry = pa.table(entry_dict)

        #             parquet_writer.write_table(table=entry)

        # parquet_writer.close()

        logging.info("Summary dataset created")

    def clear_dataset(self):
        # # Create an empty table
        # empty_table = pa.Table.from_batches([], schema=self.schema)

        # # Write empty Parquet file
        # pq.write_table(empty_table, Config.summary_dataset_filepath)
        os.remove(Config.summary_dataset_filepath)



        



