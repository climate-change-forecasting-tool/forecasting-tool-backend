from datetime import datetime
from typing import List, Literal
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
    def climate_point_query(self, longitude: float, latitude: float, start: str, end: str):
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
    
    # dates must be in "YYYYMMDD" format
    def climate_region_query(
        self, 
        longitude_min: float, 
        longitude_max: float,
        latitude_min: float, 
        latitude_max: float,
        start: str, 
        end: str
    ):
        """
        Fetch temperature, precipitation
        """
        data = self.controller.point_time_query(
            parameters=list(NASAService.climate_query_params.keys()),
            start=start,
            end=end,
            longitude_min=longitude_min,
            longitude_max=longitude_max,
            latitude_min=latitude_min,
            latitude_max=latitude_max,
            community='AG',
            time_resolution='daily')

        return data
    
    def gen_climate_point_query(self, params: List[str], longitude: float, latitude: float, start: str, end: str):
        data = self.controller.point_time_query(
            parameters=params,
            start=start,
            end=end,
            longitude=longitude,
            latitude=latitude,
            community='AG',
            time_resolution='daily')

        return data
    
    def gen_climate_region_query(
        self, 
        params: List[str], 
        longitude_min: float, 
        longitude_max: float,
        latitude_min: float, 
        latitude_max: float,
        start: str, 
        end: str
    ):
        data = self.controller.region_time_query(
            parameters=params,
            start=start,
            end=end,
            longitude_min=longitude_min,
            longitude_max=longitude_max,
            latitude_min=latitude_min,
            latitude_max=latitude_max,
            community='AG',
            time_resolution='daily')

        return data
    
    def get_weekly_point_norm_data(self, longitude: float, latitude: float, start: str, end: str):
        data = self.climate_point_query(longitude=longitude, latitude=latitude, start=start, end=end)

        parameter_data = data.get("properties", {}).get("parameter", {})

        parameter_names = list(parameter_data.keys())
        
        dates = list(parameter_data[parameter_names[0]].keys())

        coords = data.get('geometry', {}).get('coordinates') # [longitude, latitude, elevation]

        num_entries = len(dates)

        column_names = flatten_list(['timestamp', 'longitude', 'latitude', 'elevation', parameter_names])
        datalist = [
            dates, 
            [coords[0]] * num_entries, 
            [coords[1]] * num_entries, 
            [coords[2]] * num_entries
        ]
        # assuming all parameters have the same dates
        for parameter in parameter_names:
            datalist.append(list(parameter_data.get(parameter, {}).values()))
        
        df = pd.DataFrame(data=np.transpose(datalist), columns=column_names)

        df.set_index('timestamp', inplace=False)

        df[['longitude', 'latitude', 'elevation']] = df[['longitude', 'latitude', 'elevation']].astype(np.float64)
        df[parameter_names] = df[parameter_names].astype(np.float64)

        for param, (min_val, max_val) in NASAService.climate_query_params.items():
            df[param] = NASAService.minmax_scaler(
                data=df[param], 
                min_val=min_val,
                max_val=max_val
            )

        timestamp_adj = (pd.to_datetime(df['timestamp'], format='%Y%m%d') - pd.to_datetime(start, format='%Y%m%d')).dt.days.astype(np.uint64) // 7
        timestamp_adj = timestamp_adj.to_numpy().reshape(-1, 7)
        timestamp_adj = timestamp_adj[:, 0]

        df_fixed = {
            "timestamp": timestamp_adj,
            "elevation": NASAService.daily_to_weekly_conversion(df['elevation'], 'avg'),
            "avg_temperature_2m": NASAService.daily_to_weekly_conversion(df['T2M'], 'avg'),
            "min_temperature_2m": NASAService.daily_to_weekly_conversion(df['T2M_MIN'], 'min'),
            "max_temperature_2m": NASAService.daily_to_weekly_conversion(df['T2M_MAX'], 'max'),
            "avg_dewfrostpoint_2m": NASAService.daily_to_weekly_conversion(df['T2MDEW'], 'avg'),
            "min_dewfrostpoint_2m": NASAService.daily_to_weekly_conversion(df['T2MDEW'], 'min'),
            "max_dewfrostpoint_2m": NASAService.daily_to_weekly_conversion(df['T2MDEW'], 'max'),
            "avg_precipitation": NASAService.daily_to_weekly_conversion(df['PRECTOTCORR'], 'avg'),
            "min_precipitation": NASAService.daily_to_weekly_conversion(df['PRECTOTCORR'], 'min'),
            "max_precipitation": NASAService.daily_to_weekly_conversion(df['PRECTOTCORR'], 'max'),
            "avg_windspeed_2m": NASAService.daily_to_weekly_conversion(df['WS2M'], 'avg'),
            "min_windspeed_2m": NASAService.daily_to_weekly_conversion(df['WS2M_MIN'], 'min'),
            "max_windspeed_2m": NASAService.daily_to_weekly_conversion(df['WS2M_MAX'], 'max'),
            "avg_windspeed_10m": NASAService.daily_to_weekly_conversion(df['WS10M'], 'avg'),
            "min_windspeed_10m": NASAService.daily_to_weekly_conversion(df['WS10M_MIN'], 'min'),
            "max_windspeed_10m": NASAService.daily_to_weekly_conversion(df['WS10M_MAX'], 'max'),
            "avg_windspeed_50m": NASAService.daily_to_weekly_conversion(df['WS50M'], 'avg'),
            "min_windspeed_50m": NASAService.daily_to_weekly_conversion(df['WS50M_MIN'], 'min'),
            "max_windspeed_50m": NASAService.daily_to_weekly_conversion(df['WS50M_MAX'], 'max'),
            "avg_humidity_2m": NASAService.daily_to_weekly_conversion(df['RH2M'], 'avg'),
            "min_humidity_2m": NASAService.daily_to_weekly_conversion(df['RH2M'], 'min'),
            "max_humidity_2m": NASAService.daily_to_weekly_conversion(df['RH2M'], 'max'),
            "avg_surface_pressure": NASAService.daily_to_weekly_conversion(df['PS'], 'avg'),
            "min_surface_pressure": NASAService.daily_to_weekly_conversion(df['PS'], 'min'),
            "max_surface_pressure": NASAService.daily_to_weekly_conversion(df['PS'], 'max'),
            "avg_transpiration": NASAService.daily_to_weekly_conversion(df['EVPTRNS'], 'avg'),
            "min_transpiration": NASAService.daily_to_weekly_conversion(df['EVPTRNS'], 'min'),
            "max_transpiration": NASAService.daily_to_weekly_conversion(df['EVPTRNS'], 'max'),
            "avg_evaporation": NASAService.daily_to_weekly_conversion(df['EVLAND'], 'avg'),
            "min_evaporation": NASAService.daily_to_weekly_conversion(df['EVLAND'], 'min'),
            "max_evaporation": NASAService.daily_to_weekly_conversion(df['EVLAND'], 'max'),
        }

        df_fixed = pd.DataFrame(df_fixed)

        return df_fixed
    
    # def get_weekly_region_norm_data(
    #     self, 
    #     longitude_min: float, 
    #     longitude_max: float,
    #     latitude_min: float, 
    #     latitude_max: float,
    #     start: str, 
    #     end: str
    # ):
    #     data = self.climate_point_query(longitude=longitude, latitude=latitude, start=start, end=end)
    #     data = self.climate_region_query(
    #         longitude_min=longitude_min,
    #         longitude_max=longitude_max,
    #         latitude_min=latitude_min,
    #         latitude_max=latitude_max,
    #         start=start,
    #         end=end
    #     )

    #     parameter_data = data.get("properties", {}).get("parameter", {})

    #     parameter_names = list(parameter_data.keys())
        
    #     dates = list(parameter_data[parameter_names[0]].keys())

    #     coords = data.get('geometry', {}).get('coordinates') # [longitude, latitude, elevation]

    #     num_entries = len(dates)

    #     column_names = flatten_list(['timestamp', 'longitude', 'latitude', 'elevation', parameter_names])
    #     datalist = [
    #         dates, 
    #         [coords[0]] * num_entries, 
    #         [coords[1]] * num_entries, 
    #         [coords[2]] * num_entries
    #     ]
    #     # assuming all parameters have the same dates
    #     for parameter in parameter_names:
    #         datalist.append(list(parameter_data.get(parameter, {}).values()))
        
    #     df = pd.DataFrame(data=np.transpose(datalist), columns=column_names)

    #     df.set_index('timestamp', inplace=False)

    #     df[['longitude', 'latitude', 'elevation']] = df[['longitude', 'latitude', 'elevation']].astype(np.float64)
    #     df[parameter_names] = df[parameter_names].astype(np.float64)

    #     for param, (min_val, max_val) in NASAService.climate_query_params.items():
    #         df[param] = NASAService.minmax_scaler(
    #             data=df[param], 
    #             min_val=min_val,
    #             max_val=max_val
    #         )

    #     timestamp_adj = (pd.to_datetime(df['timestamp'], format='%Y%m%d') - pd.to_datetime(start, format='%Y%m%d')).dt.days.astype(np.uint64) // 7
    #     timestamp_adj = timestamp_adj.to_numpy().reshape(-1, 7)
    #     timestamp_adj = timestamp_adj[:, 0]

    #     df_fixed = {
    #         "timestamp": timestamp_adj,
    #         "elevation": NASAService.daily_to_weekly_conversion(df['elevation'], 'avg'),
    #         "avg_temperature_2m": NASAService.daily_to_weekly_conversion(df['T2M'], 'avg'),
    #         "min_temperature_2m": NASAService.daily_to_weekly_conversion(df['T2M_MIN'], 'min'),
    #         "max_temperature_2m": NASAService.daily_to_weekly_conversion(df['T2M_MAX'], 'max'),
    #         "avg_dewfrostpoint_2m": NASAService.daily_to_weekly_conversion(df['T2MDEW'], 'avg'),
    #         "min_dewfrostpoint_2m": NASAService.daily_to_weekly_conversion(df['T2MDEW'], 'min'),
    #         "max_dewfrostpoint_2m": NASAService.daily_to_weekly_conversion(df['T2MDEW'], 'max'),
    #         "avg_precipitation": NASAService.daily_to_weekly_conversion(df['PRECTOTCORR'], 'avg'),
    #         "min_precipitation": NASAService.daily_to_weekly_conversion(df['PRECTOTCORR'], 'min'),
    #         "max_precipitation": NASAService.daily_to_weekly_conversion(df['PRECTOTCORR'], 'max'),
    #         "avg_windspeed_2m": NASAService.daily_to_weekly_conversion(df['WS2M'], 'avg'),
    #         "min_windspeed_2m": NASAService.daily_to_weekly_conversion(df['WS2M_MIN'], 'min'),
    #         "max_windspeed_2m": NASAService.daily_to_weekly_conversion(df['WS2M_MAX'], 'max'),
    #         "avg_windspeed_10m": NASAService.daily_to_weekly_conversion(df['WS10M'], 'avg'),
    #         "min_windspeed_10m": NASAService.daily_to_weekly_conversion(df['WS10M_MIN'], 'min'),
    #         "max_windspeed_10m": NASAService.daily_to_weekly_conversion(df['WS10M_MAX'], 'max'),
    #         "avg_windspeed_50m": NASAService.daily_to_weekly_conversion(df['WS50M'], 'avg'),
    #         "min_windspeed_50m": NASAService.daily_to_weekly_conversion(df['WS50M_MIN'], 'min'),
    #         "max_windspeed_50m": NASAService.daily_to_weekly_conversion(df['WS50M_MAX'], 'max'),
    #         "avg_humidity_2m": NASAService.daily_to_weekly_conversion(df['RH2M'], 'avg'),
    #         "min_humidity_2m": NASAService.daily_to_weekly_conversion(df['RH2M'], 'min'),
    #         "max_humidity_2m": NASAService.daily_to_weekly_conversion(df['RH2M'], 'max'),
    #         "avg_surface_pressure": NASAService.daily_to_weekly_conversion(df['PS'], 'avg'),
    #         "min_surface_pressure": NASAService.daily_to_weekly_conversion(df['PS'], 'min'),
    #         "max_surface_pressure": NASAService.daily_to_weekly_conversion(df['PS'], 'max'),
    #         "avg_transpiration": NASAService.daily_to_weekly_conversion(df['EVPTRNS'], 'avg'),
    #         "min_transpiration": NASAService.daily_to_weekly_conversion(df['EVPTRNS'], 'min'),
    #         "max_transpiration": NASAService.daily_to_weekly_conversion(df['EVPTRNS'], 'max'),
    #         "avg_evaporation": NASAService.daily_to_weekly_conversion(df['EVLAND'], 'avg'),
    #         "min_evaporation": NASAService.daily_to_weekly_conversion(df['EVLAND'], 'min'),
    #         "max_evaporation": NASAService.daily_to_weekly_conversion(df['EVLAND'], 'max'),
    #     }

    #     df_fixed = pd.DataFrame(df_fixed)

    #     return df_fixed


    
    @staticmethod
    def json_to_dataframe(data, normalize_params: bool = False):

        parameter_data = data.get("properties", {}).get("parameter", {})

        parameter_names = list(parameter_data.keys())
        
        dates = list(parameter_data[parameter_names[0]].keys())

        coords = data.get('geometry', {}).get('coordinates') # [longitude, latitude, elevation]

        num_entries = len(dates)

        column_names = flatten_list(['timestamp', 'longitude', 'latitude', 'elevation', parameter_names])
        datalist = [
            dates, 
            [coords[0]] * num_entries, 
            [coords[1]] * num_entries, 
            [coords[2]] * num_entries
        ]
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
                    max_val=max_val
                )

        return df
    
    @staticmethod
    def minmax_scaler(data, min_val, max_val):
        return (data - min_val) / (max_val - min_val)
    
    @staticmethod
    def daily_to_weekly_conversion(data: np.ndarray, op: Literal['avg', 'max', 'min']):
        if type(data) != np.ndarray:
            data = data.to_numpy()
        converted = data.reshape(-1, 7)
        if op == 'avg':
            converted = np.mean(converted, axis=1)
        elif op == 'max':
            converted = np.max(converted, axis=1)
        elif op == 'min':
            converted = np.min(converted, axis=1)
        else:
            raise ValueError(f"Invalid operation: {op}.")
        
        return converted