import csv
from datetime import datetime
from .services import NOAAService, NASAService
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)

load_dotenv('.env')

# ns = NOAAService(token=os.getenv('NOAA_API_TOKEN'))

nasa_serv = NASAService()

data = nasa_serv.gen_climate_query(params=['T2M', 'RH2M', 'WS2M', 'PRECTOTCORR'], 
                            latitude=37.7749, 
                            longitude=-122.4194, 
                            start=datetime(2022, 1, 1), 
                            end=datetime(2025, 12, 1))

# logging.info(data)

datalist, colnames = NASAService.json_to_dataset(data)

from .model import TFTransformer

tft = TFTransformer(datalist=datalist, column_names=colnames)


