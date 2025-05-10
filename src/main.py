import csv
from datetime import datetime

from .configuration.config import Config
import logging

from .datamodels.point_generation_model import SummaryDataset

logging.basicConfig(level=logging.INFO)

"""
EM-DAT: disaster data
    - https://doc.emdat.be/docs/legal/terms-of-use/
NASA Earthdata | GDIS: geocoded disaster data
NASA POWER API: weather data

"""

if Config.generate_summary_dataset:
    sds = SummaryDataset()

if Config.activate_tft:
    from .model import TFTransformer

    tft = TFTransformer()