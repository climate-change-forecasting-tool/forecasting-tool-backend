import csv
from datetime import datetime

from .services import NOAAService, NASAService
from .configuration.config import Config
import os
from dotenv import load_dotenv
import logging

from .dataset_summary import SummaryDataset

logging.basicConfig(level=logging.INFO)

load_dotenv('.env')
"""
EM-DAT: disaster data
    - https://doc.emdat.be/docs/legal/terms-of-use/
NASA Earthdata | GDIS: geocoded disaster data
NASA POWER API: weather data

"""

sds = SummaryDataset()

if Config.generate_summary_dataset:
    if Config.recreate_summary_dataset:
        sds.clear_dataset()

    sds.upload_data(start_date=Config.start_date, end_date=Config.end_date)

if Config.activate_tft:
    from .model import TFTransformer

    tft = TFTransformer(parquet_df='db/summary_data.parquet')


