from datetime import datetime
import os
from typing import List, Tuple
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

from src.configuration.config import Config
from src.utility import get_astronomical_season_df

import logging
logging.basicConfig(level=logging.INFO)

class SummaryDataset:
    def __init__(self):
        pass

        self.climate_params = [
            ('t2m_min', pa.float32()),
            ('t2m_mean', pa.float32()),
            ('t2m_max', pa.float32()),
            ('u10_min', pa.float32()),
            ('u10_mean', pa.float32()),
            ('u10_max', pa.float32()),
            ('v10_min', pa.float32()),
            ('v10_mean', pa.float32()),
            ('v10_max', pa.float32()),
            ('lsm_mean', pa.float32()),
            ('sp_mean', pa.float32()),
            ('aod550_max', pa.float32()),
            ('tc_ch4_max', pa.float32()),
            ('tcno2_max', pa.float32()),
            ('gtco3_max', pa.float32()),
            ('tcso2_max', pa.float32()),
            ('tcwv_max', pa.float32()),
        ]

        self.schema = pa.schema([
            ("timestamp", pa.date64()),
            ("group_id", pa.int64()),
            ("longitude", pa.float64()),
            ("latitude", pa.float64()),
            ("season", pa.string()),
        ] + self.climate_params)

        self.table_creation()

    def table_creation(self):
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

    def generate(self, indexes: List[int], points: List[Tuple[float, float]]):

        try:
            group_point_df = pd.DataFrame(
                data=np.hstack([np.array(points), np.array(indexes)[:, np.newaxis]]), 
                columns=["longitude", "latitude", "group_id"]
            ).set_index(['group_id'])

            dates = pd.date_range(start=Config.start_date, end=Config.end_date, freq="MS", inclusive="both")

            parquet_writer = pq.ParquetWriter(where=Config.summary_dataset_filepath, schema=self.schema)

            for date in dates:
                logging.info(f"Concatenating files for month of {date}")
                small_combined_df = pd.concat([
                    pd.read_parquet(
                        path=Config.partitioned_climate_data_dir + var_name + '_' + str(date.year) + '_' + str(date.month) + '.parquet', 
                        engine="pyarrow"
                    ).set_index(['date', 'group_id'])
                    for var_name in list(Config.climate_data_param_names.keys())
                ], axis=1)

                point_combined_df = small_combined_df.join(other=group_point_df)

                point_combined_df.reset_index(drop=False, inplace=True)
                point_combined_df.rename(columns={'date': 'timestamp'}, inplace=True)
                
                point_combined_df["season"] = point_combined_df.apply(get_astronomical_season_df, axis=1)

                # logging.info(small_combined_df)

                batch_dict = dict()
                batch_dict.update({
                    "timestamp": pa.array(point_combined_df['timestamp'], type=pa.date64()),
                    "group_id": pa.array(point_combined_df['group_id'], type=pa.int64()),
                    "longitude": pa.array(point_combined_df['longitude'], type=pa.float64()),
                    "latitude": pa.array(point_combined_df['latitude'], type=pa.float64()),
                    "season": pa.array(point_combined_df['season'], type=pa.string()),
                })

                for var_name in self.climate_params:
                    batch_dict.update({
                        var_name[0]: pa.array(point_combined_df[var_name[0]], type=var_name[1]),
                    })

                batch_data = pa.table(batch_dict)

                parquet_writer.write_table(batch_data)

            parquet_writer.close()

            logging.info("Summary dataset created")

            logging.info(pd.read_parquet(path="db/summary_data.parquet", engine="pyarrow"))
        except Exception as e:
            logging.error(e)

    def clear_dataset(self):
        empty_table = pa.Table.from_batches([], schema=self.schema)

        # Write empty Parquet file
        pq.write_table(empty_table, Config.summary_dataset_filepath)




        



