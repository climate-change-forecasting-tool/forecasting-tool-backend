import os
from typing import List, Tuple
import numpy as np
import pyarrow as pa
import pandas as pd

from src.configuration.config import Config

import logging
logging.basicConfig(level=logging.INFO)

class SummaryDataset:
    def __init__(self):

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

    def generate(self, indexes: List[int], points: List[Tuple[float, float]]):
        if os.path.exists(Config.summary_dataset_filepath):
            if Config.recreate_summary_dataset:
                logging.info("Clearing Summary Dataset Parquet file and recreating now...")
                self.clear_dataset()
            else:
                logging.info("Skipping Summary Dataset Parquet file creation.")
                return
        else:
            logging.info("Summary Dataset Parquet file does not exist. Creating now...")

        partitioned_files = [f for f in os.listdir(Config.partitioned_climate_data_dir)]
        file_variables = [v[:-5] for v in partitioned_files]

        group_point_df = pd.DataFrame(
            data=np.hstack([np.array(points), np.array(indexes)[:, np.newaxis]]), 
            columns=["target_lon", "target_lat", "group_id"]
        ).set_index(['group_id'])

        total_df = pd.concat([
            pd.read_parquet(
                path=Config.partitioned_climate_data_dir + partitioned_file, 
                engine="pyarrow"
            ).set_index(['date', 'group_id'])
            for partitioned_file in partitioned_files
        ], axis=1)

        logging.info(total_df)

        final_df = total_df.join(other=group_point_df)

        logging.info(final_df)

        final_df.reset_index(drop=False, inplace=True)
        final_df.rename(columns={'date': 'timestamp'}, inplace=True)

        final_df.to_parquet(path=Config.summary_dataset_filepath, engine='pyarrow', index=True)

        logging.info("Summary dataset created")

    def clear_dataset(self):
        os.remove(Config.summary_dataset_filepath)



        



