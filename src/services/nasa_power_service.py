from datetime import datetime
from typing import List
from src.controllers.nasa_power_controller import NASAController
from src.utility import flatten_list
import numpy as np
import pandas as pd

# TODO: predict long-term features! use minimum, maximum, and average of monthly data

# https://power.larc.nasa.gov/beta/parameters/

class NASAService:
    climate_query_params = [
        'T2M', # temperature at 2 meters
        'T2M_MIN',
        'T2M_MAX',
        'T2MDEW', # dew/frost point at 2 meters
        'PRECTOTCORR',
        'WS2M', # wind speed at 2 meters
        'WS2M_MIN',
        'WS2M_MAX',
        'WS10M', # wind speed at 10 meters
        'WS10M_MIN',
        'WS10M_MAX',
        'WS50M', # wind speed at 50 meters
        'WS50M_MIN',
        'WS50M_MAX',
        'RH2M', # relative humidity at 2 meters
        'PS', # surface pressure
        'EVPTRNS', # transpiration
        'EVLAND', # evaporation
    ]

    # climate_query_params = [
    #     'T2M', # temperature at 2 meters
    #     'RH2M', # relative humidity at 2 meters
    #     'T2MDEW', # dew/frost point at 2 meters
    #     'PS', # surface pressure
    #     # 'RHOA', # surface air density
    #     'PRECTOTCORR', # precipitation corrected
    #     'PRECSNO', # precipitation snow
    #     'CLOUD_AMT', # cloud amount
    #     # WIND STUFF
    #     'WS2M', # wind speed at 2 meters
    #     'WD2M', # wind direction at 2 meters
    #     'WS10M', # wind speed at 10 meters
    #     'WD10M', # wind direction at 10 meters
    #     'WS50M', # wind speed at 50 meters
    #     'WD50M', # wind direction at 50 meters
    #     # SUN STUFF (will predict seasonal stuff)
    #     # 'SZA', # solar zenith angle
    #     # 'ALLSKY_SFC_SW_DWN', # All Sky Surface Shortwave Downward Irradiance
    #     # 'CLRSKY_SFC_SW_DWN', # Clear Sky Surface Shortwave Downward Irradiance
    #     # 'ALLSKY_SFC_UVA', # All Sky Surface UVA Irradiance
    #     # 'PSH', # Peak sun hour

    #     # LAND STUFF
    #     # 'Z0M', # Surface Roughness

    #     # SEA STUFF
    #     # 'SLP', # Sea level pressure

    #     # AIR STUFF
    #     # 'AIRMASS', # Air mass
    #     # 'PBLTOP', # planetary boundary layer top pressure

    #     # ATMOSPHERE STUFF
    #     # 'TO3', # total column ozone 

    #     # DROUGHT STUFF
    #     'EVPTRNS', # transpiration
    #     'EVLAND', # evaporation
    # ]

    def __init__(self):
        self.controller = NASAController()

    # climate-stuff
    # dates must be in "YYYYMMDD" format
    def climate_query(self, longitude: float, latitude: float, start: str, end: str):
        """
        Fetch temperature, precipitation
        """
        data = self.controller.point_time_query(
            parameters=NASAService.climate_query_params,
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
    def json_to_dataframe(data):

        parameter_data = data.get("properties", {}).get("parameter", {})

        parameter_names = list(parameter_data.keys())
        # dates = list(int(item) for item in parameter_data[parameter_names[0]].keys())
        dates = list(parameter_data[parameter_names[0]].keys())

        coords = data.get('geometry', {}).get('coordinates') # [longitude, latitude, elevation]

        num_entries = len(dates)

        column_names = flatten_list(['timestamp', 'longitude', 'latitude', 'elevation', parameter_names])
        datalist = [dates, [coords[0]] * num_entries, [coords[1]] * num_entries, [coords[2]] * num_entries]
        # assuming all parameters have the same dates
        for parameter in parameter_names:
            datalist.append(list(parameter_data.get(parameter, {}).values()))
        
        df = pd.DataFrame(data=np.transpose(datalist), columns=column_names)

        df.set_index('timestamp', inplace=True)

        df[['longitude', 'latitude', 'elevation']] = df[['longitude', 'latitude', 'elevation']].astype('float64')
        df[parameter_names] = df[parameter_names].astype('float32')

        return df
    



# with open("weatherdata.csv", mode="w", newline="") as file:
#     parameter_data = data.get("properties", {}).get("parameter", {})

#     parameter_names = list(parameter_data.keys())
#     dates = parameter_data[parameter_names[0]].keys()

#     # elevation is the 3rd item and is in meters
#     coords = data.get('geometry', {}).get('coordinates')

#     writer = csv.writer(file)
#     writer.writerow(
#         flatten_list(['timestamp', 'latitude', 'longitude', 'elevation', parameter_names])
#     ) # CSV header

#     # Write the date, hour, and temperature to the CSV file
#     for date in dates:
#         parameter_list = flatten_list([int(date), coords])
#         for parameter_name in parameter_names:
#             parameter_list.append(parameter_data.get(parameter_name, {}).get(date, -1))
#         writer.writerow(parameter_list)