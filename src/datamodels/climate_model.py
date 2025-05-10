from typing import List
import pandas as pd
import xarray as xr
from src.configuration.config import Config

# https://confluence.ecmwf.int/display/CKB/CAMS%3A+Reanalysis+data+documentation#CAMS:Reanalysisdatadocumentation-KnownissuesCAMSglobalreanalysis(EAC4)

class GRIB_Reader:
    def __init__(self, filename, variable_names):
        # Load GRIB file
        self.ds = xr.open_dataset(filename, engine="cfgrib")

        self.variable_names = variable_names

        # print("Dataset:")
        # print(self.ds)
        # print("Variables:")
        # print(self.ds.variables)

    def read_timeseries(self, longitude: float, latitude: float):
        point_data = self.ds.sel({'latitude': latitude, 'longitude': longitude}, method='nearest')

        total_df = pd.concat([
            self.convert_var_to_df(
                point_data=point_data,
                var_name=var_name
            )
            for var_name in self.variable_names
        ], axis=1)

        return total_df

    def convert_var_to_df(self, point_data: xr.Dataset, var_name: str):
        ts = point_data[var_name].to_dataframe()
        ts = ts.reset_index()

        ts.drop(['number', 'step', 'valid_time'], axis=1, inplace=True)

        ts['datetime'] = ts['time']
        ts['date'] = ts['datetime'].dt.date
        ts['time'] = ts['datetime'].dt.time
        ts.drop(['datetime'], axis=1, inplace=True)

        # check if 'surface' is significant at all

        ts = ts.groupby('date')[var_name].agg(['min', 'mean', 'max']).reset_index()

        ts['date'] = pd.to_datetime(ts['date'])

        ts.set_index(['date'], inplace=True)

        for column in ts.columns:
            ts.rename(columns={column: var_name + '_' + column}, inplace=True)

        return ts
    

class GRIB_Controller:
    def __init__(self):
        self.readers: List[GRIB_Reader] = []

        for file, vars in Config.cams_files_and_vars.items():
            self.readers.append(
                GRIB_Reader(
                    filename=file,
                    variable_names=vars
                )
            )

    def get_point_data(self, longitude: float, latitude: float):
        dates = pd.date_range(start=Config.start_date, end=Config.end_date, freq="D", inclusive="both")

        total_df = pd.DataFrame({"date": dates})

        # total_df['date'] = total_df['date'].dt.date

        total_df['date'] = pd.to_datetime(total_df['date'])

        total_df.set_index(keys=['date'], inplace=True)

        # print("Total df")
        # print(total_df)

        if longitude < 0:
            longitude += 360

        for reader in self.readers:
            total_df = pd.concat([
                total_df,
                reader.read_timeseries(
                    longitude=longitude,
                    latitude=latitude
                )
                ], axis=1
            )

        return total_df