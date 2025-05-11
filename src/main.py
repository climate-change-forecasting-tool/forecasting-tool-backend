import csv
from datetime import datetime

from .services import NASAService
from .configuration.config import Config
import logging

from .dataset_summary import SummaryDataset

logging.basicConfig(level=logging.INFO)

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

    tft = TFTransformer()