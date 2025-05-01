import csv
from datetime import datetime

from .services import NASAService
from .configuration.config import Config
import logging

logging.basicConfig(level=logging.INFO)

"""
EM-DAT: disaster data
    - https://doc.emdat.be/docs/legal/terms-of-use/
NASA Earthdata | GDIS: geocoded disaster data
NASA POWER API: weather data

"""


from .model import TFTransformer

tft = TFTransformer()


