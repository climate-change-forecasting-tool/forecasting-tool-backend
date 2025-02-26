from datetime import datetime
import time
from typing import List, Literal
import requests
import logging

logging.basicConfig(level=logging.INFO)


MAX_RETRIES = 5

# https://power.larc.nasa.gov/docs/services/api/

class NASAController:
    def __init__(self):
        self.entry_url = "https://power.larc.nasa.gov/api"

    def execute_query(self, endpoint: str, params: dict):
        next_retry_time = 1
        next_retry_multi = 2
        retry = 0
        good_response = True
        while good_response and retry < MAX_RETRIES:
            response = requests.get(url=self.entry_url + endpoint, 
                                    params=params)
            response.close()
            match(response.status_code):
                case 200: # accepted
                    # the response's data is JSON
                    response_data = response.json()
                    return response_data
                case 429: 
                    logging.info(f"NASA POWER request limit reached {retry+1}. Retrying in {next_retry_time} seconds")
                    time.sleep(next_retry_time)
                    next_retry_time *= next_retry_multi
                    break
                case _:
                    logging.info(f"NASA POWER API error: {response.status_code}")
                    good_response = False
                    break
            retry += 1
        return {}
    
    def point_time_query(self, parameters: List[str], start: str, end: str, longitude: float, latitude: float, community: Literal['AG', 'SB', 'RB'], time_resolution: Literal['hourly', 'daily', 'monthly', 'climatology']):
        if datetime.strptime(end, "%Y%m%d") < datetime.strptime(start, "%Y%m%d") or datetime.strptime(start, "%Y%m%d") > datetime.now():
            print(start)
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
            "latitude": latitude,
            "longitude": longitude,
            "community": community,
            "format": "JSON" # JSON
        }

        # TODO: split into monthly queries then compile
        data = self.execute_query(endpoint=f'/temporal/{time_resolution}/point', params=params)

        return data
        