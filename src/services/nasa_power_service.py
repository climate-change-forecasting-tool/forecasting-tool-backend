from datetime import datetime
from typing import List
from src.controllers.nasa_power_controller import NASAController
from src.utility import flatten_list
import numpy as np
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

# TODO: predict long-term features! use minimum, maximum, and average of monthly data

# https://power.larc.nasa.gov/beta/parameters/

class NASAService:
    climate_query_params = {
        'T2M': (-125, 80), # temperature at 2 meters
        'T2M_MIN': (-125, 80),
        'T2M_MAX': (-125, 80),
        'T2MDEW': (-125, 80), # dew/frost point at 2 meters
        'PRECTOTCORR': (0, 12000),
        'WS2M': (0, 50), # wind speed at 2 meters
        'WS2M_MIN': (0, 50),
        'WS2M_MAX': (0, 50),
        'WS10M': (0, 50), # wind speed at 10 meters
        'WS10M_MIN': (0, 50),
        'WS10M_MAX': (0, 50),
        'WS50M': (0, 75), # wind speed at 50 meters
        'WS50M_MIN': (0, 75),
        'WS50M_MAX': (0, 75),
        'RH2M': (0, 100), # relative humidity at 2 meters
        'PS': (50, 110), # surface pressure
        'EVPTRNS': (0, 5.40), # transpiration
        'EVLAND': (-500, 500), # evaporation
    }

    def __init__(self):
        self.controller = NASAController()

    # climate-stuff
    # dates must be in "YYYYMMDD" format
    def climate_query(self, longitude: float, latitude: float, start: str, end: str):
        """
        Fetch temperature, precipitation
        """
        data = self.controller.point_time_query(
            parameters=list(NASAService.climate_query_params.keys()),
            start=start,
            end=end,
            longitude=longitude,
            latitude=latitude,
            community='AG',
            time_resolution='daily')

        return data
    
    def gen_climate_query(self, params: List[str], longitude: int, latitude: int, start: str, end: str):
        data = self.controller.point_time_query(
            parameters=params,
            start=start,
            end=end,
            longitude=longitude,
            latitude=latitude,
            community='AG',
            time_resolution='daily')

        return data
    
    @staticmethod
    def json_to_dataframe(data, normalize_params: bool = False):

        parameter_data = data.get("properties", {}).get("parameter", {})

        parameter_names = list(parameter_data.keys())
        # dates = list(int(item) for item in parameter_data[parameter_names[0]].keys())
        dates = list(parameter_data[parameter_names[0]].keys())

        # TODO: double check geometry
        coords = data.get('geometry', {}).get('coordinates') # [longitude, latitude, elevation]

        num_entries = len(dates)

        column_names = flatten_list(['timestamp', 'longitude', 'latitude', 'elevation', parameter_names])
        datalist = [dates, [coords[0]] * num_entries, [coords[1]] * num_entries, [coords[2]] * num_entries]
        # assuming all parameters have the same dates
        for parameter in parameter_names:
            datalist.append(list(parameter_data.get(parameter, {}).values()))
        
        df = pd.DataFrame(data=np.transpose(datalist), columns=column_names)

        df.set_index('timestamp', inplace=False)

        df[['longitude', 'latitude', 'elevation']] = df[['longitude', 'latitude', 'elevation']].astype(np.float64)
        df[parameter_names] = df[parameter_names].astype(np.float64)

        # TODO: assumption that this is the main data climate query thing
        if normalize_params:
            for param, (min_val, max_val) in NASAService.climate_query_params.items():
                df[param] = NASAService.minmax_scaler(
                    data=df[param], 
                    min_val=min_val,
                    max_val=max_val)
                
            df['longitude'] = (df['longitude'] + 180.0) / 360.0 # TODO: not needed
            df['latitude'] = (df['latitude'] + 90.0) / 180.0 # TODO: not needed
            df['elevation'] = df['elevation'] / 8000.0

        return df
    
    def minmax_scaler(data, min_val, max_val): # TODO: put elsewhere
        return (data - min_val) / (max_val - min_val)