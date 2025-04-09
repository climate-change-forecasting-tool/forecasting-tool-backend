import logging
import geopandas as gpd
import pandas as pd
from src.configuration.config import Config

logging.basicConfig(level=logging.INFO)

class DisasterModel:
    def __init__(self):
        pass

    def get_table(self):
        gdis_df = gpd.read_file( # TODO: evaluate each disaster's geometry mapped to explain why there is duplicate data & other stuff
            filename=Config.disaster_gdb_filepath, 
            columns=['disasterno', 'iso3', 'disastertype', 'geometry'],
            ignore_geometry=False
        ) # 39952 entries

        # disastertypes: 'landslide', 'flood', 'volcanic activity', 'earthquake', 'mass movement (dry)', 'extreme temperature ', 'storm', 'drought'

        # exclude volcanoes & earthquakes
        gdis_df.drop(gdis_df[gdis_df['disastertype'].isin(['volcanic activity', 'earthquake'])].index, inplace=True)

        # filter out entries that have null iso3's
        gdis_df = gdis_df.loc[gdis_df['iso3'].notnull()] # TODO: this might need to be fixed

        # TODO: store each geometry's centroid as a longitude and latitude for faster distance calculations?

        # add iso3 to disasterno to get the corresponding EM-DAT disaster number
        gdis_df['disasterno_iso3'] = gdis_df['disasterno'] + '-' + gdis_df['iso3']

        # drop iso3
        gdis_df.drop('iso3', axis=1, inplace=True)

        gdis_df.drop_duplicates(keep='first', inplace=True) # 9780

        # print(gdis_df)

        # TODO: maybe include no. affected / total no. affected
        emdat_df = pd.read_excel(
            io=Config.emdat_data_filepath,
            usecols=['DisNo.', 'Start Year', 'Start Month', 'Start Day', 'End Year', 'End Month', 'End Day', 'Total Deaths', 'No. Injured', "Total Damage, Adjusted ('000 US$)"]
        ) # 26090 entries

        # rename disaster name
        emdat_df.rename(columns={'DisNo.': 'disasterno_iso3', 'Total Deaths': 'total_deaths', 'No. Injured': 'num_injured', "Total Damage, Adjusted ('000 US$)": 'damage_cost'}, inplace=True)

        # filter out entries that have empty start dates or end dates
        emdat_df = emdat_df[(emdat_df['Start Month'].notnull()) & (emdat_df['Start Day'].notnull()) & (emdat_df['End Month'].notnull()) & (emdat_df['End Day'].notnull())]

        emdat_df[['Start Year', 'Start Month', 'Start Day', 'End Year', 'End Month', 'End Day']] = \
            emdat_df[['Start Year', 'Start Month', 'Start Day', 'End Year', 'End Month', 'End Day']].astype(int).astype(str)

        emdat_df['start_date'] = emdat_df['Start Year'].str.zfill(4) + emdat_df['Start Month'].str.zfill(2) + emdat_df['Start Day'].str.zfill(2)
        emdat_df['end_date'] = emdat_df['End Year'].str.zfill(4) + emdat_df['End Month'].str.zfill(2) + emdat_df['End Day'].str.zfill(2)

        emdat_df.drop(['Start Year', 'Start Month', 'Start Day', 'End Year', 'End Month', 'End Day'], axis=1, inplace=True) # 22105

        # print(emdat_df)

        # TODO: ensure that we retrieve as much data as possible from this
        combined_df = pd.merge(left=emdat_df, right=gdis_df, on='disasterno_iso3', how='inner') # 7326

        combined_df.fillna(0, inplace=True)

        # logging.info(combined_df.columns)

        combined_df[['total_deaths', 'num_injured', 'damage_cost']] = combined_df[['total_deaths', 'num_injured', 'damage_cost']].astype(int)
        combined_df['start_date'] = pd.to_datetime(combined_df['start_date'])

        # TODO: drop all entries preceding the furthest back point
        combined_df['end_date'] = pd.to_datetime(combined_df['end_date'])
        combined_df = combined_df[combined_df['end_date'] >= Config.furthest_back_time]

        return combined_df