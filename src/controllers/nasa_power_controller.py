from datetime import datetime
import time
from typing import List, Literal, Tuple
import requests
import logging
from requests.exceptions import RequestException

logging.basicConfig(level=logging.INFO)


MAX_RETRIES = 5

# https://power.larc.nasa.gov/docs/services/api/

class NASAController:
    def __init__(self):
        self.entry_url = "https://power.larc.nasa.gov/api"
        self.session = requests.Session()

    def execute_query(self, endpoint: str, params: dict):
        next_retry_time = 5
        next_retry_multi = 2
        retry = 0
        while retry < MAX_RETRIES:
            try:
                response = self.session.get(url=self.entry_url + endpoint, params=params)
                response.close()
            except RequestException as e:
                logging.info(f"RequestException encountered: {e}. Retrying in {next_retry_time} seconds...")
                time.sleep(next_retry_time)
                next_retry_time *= next_retry_multi
                retry += 1
                continue

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                logging.info(f"NASA POWER request limit reached {retry+1}. Retrying in {next_retry_time} seconds")
                time.sleep(next_retry_time)
                next_retry_time *= next_retry_multi
                retry += 1
            else:
                logging.info(f"NASA POWER API error: {response.status_code}")
                break

        return {}

    
    def point_time_query(
        self, 
        parameters: List[str], 
        start: str, 
        end: str, 
        longitude: float, 
        latitude: float, 
        community: Literal['AG', 'SB', 'RB'], 
        time_resolution: Literal['hourly', 'daily', 'monthly', 'climatology']
    ):
        if datetime.strptime(end, "%Y%m%d") < datetime.strptime(start, "%Y%m%d") or datetime.strptime(start, "%Y%m%d") > datetime.now():
            raise ValueError('The start and end datetimes are incorrectly configured.')

        if longitude < -180 or longitude > 180 or latitude < -90 or latitude > 90:
            raise ValueError('Longitude must be between -180 and 180, and latitude must be between -90 and 90.')

        if community not in ['AG', 'SB', 'RE']: # Agriculture, Sustainable Buildings, Renewable Energy
            raise ValueError('The community parameter must be either AG, SB, or RE.')
        
        if time_resolution not in ['climatology', 'monthly', 'daily', 'hourly']:
            raise ValueError('time_resolution must be one of the following: climatology, monthly, daily, or hourly.')

        params = {
            "parameters": ','.join(parameters),
            "start": start,
            "end": end,
            "longitude": longitude,
            "latitude": latitude,
            "community": community,
            "format": "JSON"
        }

        data = self.execute_query(endpoint=f'/temporal/{time_resolution}/point', params=params)

        return data
    
    def region_time_query(
        self,
        parameters: List[str], 
        start: str, 
        end: str, 
        longitude_min: float,
        longitude_max: float,
        latitude_min: float,
        latitude_max: float,
        community: Literal['AG', 'SB', 'RB'], 
        time_resolution: Literal['hourly', 'daily', 'monthly', 'climatology']
    ):
        if datetime.strptime(end, "%Y%m%d") < datetime.strptime(start, "%Y%m%d") or datetime.strptime(start, "%Y%m%d") > datetime.now():
            raise ValueError('The start and end datetimes are incorrectly configured.')
        
        if longitude_min < -180 or longitude_max > 180 or latitude_min < -90 or latitude_max > 90:
            raise ValueError('Longitude must be between -180 and 180, and latitude must be between -90 and 90.')

        if community not in ['AG', 'SB', 'RE']: # Agriculture, Sustainable Buildings, Renewable Energy
            raise ValueError('The community parameter must be either AG, SB, or RE.')
        
        if time_resolution not in ['climatology', 'monthly', 'daily', 'hourly']:
            raise ValueError('time_resolution must be one of the following: climatology, monthly, daily, or hourly.')

        params = {
            "parameters": ','.join(parameters),
            "start": start,
            "end": end,
            "longitude-min": longitude_min,
            "longitude-max": longitude_max,
            "latitude-min": latitude_min,
            "latitude-max": latitude_max,
            "community": community,
            "format": "JSON"
        }

        data = self.execute_query(endpoint=f'/temporal/{time_resolution}/region', params=params)

        return data
        