from datetime import datetime
from typing import List
from src.controllers.nasa_power_controller import NASAController
from src.utility import flatten_list
import numpy as np

# TODO: predict long-term features! use minimum, maximum, and average of monthly data

# https://power.larc.nasa.gov/beta/parameters/

class NASAService:
    climate_query_params = [
        'T2M', # temperature at 2 meters
        'RH2M', # relative humidity at 2 meters
        'T2MDEW', # dew/frost point at 2 meters
        'PS', # surface pressure
        'RHOA', # surface air density
        'PRECTOTCORR', # precipitation corrected
        'PRECSNO', # precipitation snow
        'CLOUD_AMT', # cloud amount
        # WIND STUFF
        'WS2M', # wind speed at 2 meters
        'WD2M', # wind direction at 2 meters
        'WS10M', # wind speed at 10 meters
        'WD10M', # wind direction at 10 meters
        'WS50M', # wind speed at 50 meters
        'WD50M', # wind direction at 50 meters
        # SUN STUFF (will predict seasonal stuff)
        'SZA', # solar zenith angle
        'ALLSKY_SFC_SW_DWN', # All Sky Surface Shortwave Downward Irradiance
        'CLRSKY_SFC_SW_DWN', # Clear Sky Surface Shortwave Downward Irradiance
        'ALLSKY_SFC_UVA', # All Sky Surface UVA Irradiance
        'PSH', # Peak sun hour

        # LAND STUFF
        'Z0M', # Surface Roughness

        # SEA STUFF
        'SLP', # Sea level pressure

        # AIR STUFF
        'AIRMASS', # Air mass
        'PBLTOP', # planetary boundary layer top pressure

        # ATMOSPHERE STUFF
        'TO3', # total column ozone 

        # DROUGHT STUFF
        'EVPTRNS', # transpiration
        'EVAP', # evaporation
        'EVPTOT' # total evapotranspiration
    ]

    def __init__(self):
        self.controller = NASAController()

    # climate-stuff
    # dates must be in "YYYYMMDD" format
    def climate_query(self, longitude: float, latitude: float, start: datetime, end: datetime):
        """
        Fetch temperature, precipitation
        """
        data = self.controller.point_daily_query(
            parameters=NASAService.climate_query_params,
            start=start,
            end=end,
            longitude=longitude,
            latitude=latitude,
            community='AG')

        return data
    
    def gen_climate_query(self, params: List[str], longitude: int, latitude: int, start: datetime, end: datetime):
        data = self.controller.point_daily_query(
            parameters=params,
            start=start,
            end=end,
            longitude=longitude,
            latitude=latitude,
            community='AG',
            time_resolution='hourly')

        return data
    
    @staticmethod
    def json_to_dataset(data):

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

        return np.transpose(datalist), column_names
    



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