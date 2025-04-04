import csv
from datetime import datetime

from src.point_generation_service import PointGenerator
from .services import NOAAService, NASAService, DisasterDBService, LandQueryService
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

# ddbs = DisasterDBService()

# import fiona 
# print(fiona.supported_drivers)

# lqs = LandQueryService()

# lqs.dostuff()



# pgs = PointGenerator()


sds = SummaryDataset()

if Config.generate_summary_dataset:
    if Config.recreate_summary_dataset:
        sds.clear_dataset()

    sds.upload_data(start_date=Config.start_date, end_date=Config.end_date) # 2018

# ddbs = DisasterDBService()

# result_df = ddbs.world_query(start_time='20100101', end_time='20110101')

# print(result_df)

# ns = NOAAService(token=os.getenv('NOAA_API_TOKEN'))

# nasa_serv = NASAService()

# df = NASAService.json_to_dataframe(
#     nasa_serv.climate_query(
#         longitude=-122.4194,
#         latitude=12.7749,
#         start=datetime(1990, 1, 1),
#         end=datetime(1991, 1, 1)
#     )
# )

# logging.info(df)

# data = nasa_serv.gen_climate_query(params=['T2M', 'RH2M', 'WS2M', 'PRECTOTCORR'], 
#                             latitude=37.7749, 
#                             longitude=-122.4194, 
#                             start=datetime(2022, 1, 1), 
#                             end=datetime(2023, 12, 1))

# # logging.info(data)

# df = NASAService.json_to_dataframe(data)

# logging.info(df)

# from .model import TFTransformer

# tft = TFTransformer(parquet_df='db/summary_data.parquet')


