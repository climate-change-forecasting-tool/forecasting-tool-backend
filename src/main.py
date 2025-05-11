import csv
from datetime import datetime

from .configuration.config import Config
import logging

from .datamodels import PointGenerationModel, SummaryDataset

logging.basicConfig(level=logging.INFO)

"""
EM-DAT: disaster data
    - https://doc.emdat.be/docs/legal/terms-of-use/
NASA Earthdata | GDIS: geocoded disaster data
NASA POWER API: weather data

"""

import os
if Config.download_climate_data or Config.generate_summary_dataset:

    pgm = PointGenerationModel()
    if os.path.exists("db/h3_idxs.pkl"):
        logging.info("Retrieving previous ids")
        int_h3_indexes = pgm.load_ids()
        h3_points = [pgm.get_coordinate_from_id(int_h3_index) for int_h3_index in int_h3_indexes]
    else:
        logging.info("Creating new points and ids, and saving ids")
        int_h3_indexes, h3_points = pgm.get_result()
        pgm.save_ids(int_h3_indexes)

if Config.download_climate_data:
    from .data_sourcing.downloading import Downloader

    downloader = Downloader()

    downloader.run(indexes=int_h3_indexes, points = h3_points)

if Config.generate_summary_dataset:
    sds = SummaryDataset()

    sds.generate(indexes=int_h3_indexes, points = h3_points)

if Config.activate_tft:
    from .model import TFTransformer

    tft = TFTransformer()