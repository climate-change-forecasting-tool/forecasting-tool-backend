
import sqlite3
import json
import os
from typing import List
import logging
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, MultiPolygon, Polygon
from shapely import wkb, wkt
import time
from src.configuration.config import Config

logging.basicConfig(level=logging.INFO)

DB_FILE_PATH = "data/disasters.db"

class DisasterDBController:
    def __init__(self, db_file_path = DB_FILE_PATH):
        if db_file_path is None:
            self.db_file_path = DB_FILE_PATH

        self.db_file_path = db_file_path
        self.create_table()
    
    def create_table(self):

        if Config.recreate_disaster_database:
            self.clear_all()

        elif self.check_table_exists():
            logging.info("Skipping disaster db creation!")
            return

        logging.info("Creating disaster db...")
        gdis_df = gpd.read_file( # TODO: evaluate each disaster's geometry mapped to explain why there is duplicate data & other stuff
            filename="data/pend-gdis-1960-2018-disasterlocations-gdb/pend-gdis-1960-2018-disasterlocations.gdb", 
            columns=['disasterno', 'iso3', 'disastertype', 'geometry'],
            ignore_geometry=False
        ) # 39952 entries

        # disastertypes: 'landslide', 'flood', 'volcanic activity', 'earthquake', 'mass movement (dry)', 'extreme temperature ', 'storm', 'drought'

        # exclude volcanoes & earthquakes
        gdis_df.drop(gdis_df[gdis_df['disastertype'].isin(['volcanic activity', 'earthquake'])].index, inplace=True)

        # print(gdis_df)

        # filter out entries that have null iso3's
        gdis_df = gdis_df.loc[gdis_df['iso3'].notnull()] # TODO: this might need to be fixed

        # TODO: store each geometry's centroid as a longitude and latitude for faster distance calculations?

        # add iso3 to disasterno to get the corresponding EM-DAT disaster number
        gdis_df['disasterno_iso3'] = gdis_df['disasterno'] + '-' + gdis_df['iso3']

        # drop iso3
        gdis_df.drop('iso3', axis=1, inplace=True)

        gdis_df.drop_duplicates(keep='first', inplace=True) # 9780

        # print(gdis_df)

        emdat_df = pd.read_excel(
            io='data/emdat_disasterdata.xlsx',
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

        combined_df[['total_deaths', 'num_injured', 'damage_cost']] = combined_df[['total_deaths', 'num_injured', 'damage_cost']].astype(int)

        # write back
        combined_df.to_file(data='data/disasters.gdb', driver='OpenFileGDB') # TODO: FIX

        # TODO: it might be wiser to keep this dataframe in-memory??? 1 SQL db for everything except geometry, 1 in-memory df with disaster no. and geometry
        combined_df['geometry'] = [wkt.dumps(geom) for geom in combined_df['geometry']]

        with sqlite3.connect(self.db_file_path) as conn:
            cur = conn.cursor()
            combined_df.to_sql('disasters', conn, if_exists='replace', index=False)
            conn.commit()
            cur.close()

        logging.info("table creation complete!")
    
    def get_geometry_df(self, num_rows: int = None):
        gdis_df = gpd.read_file(
            filename="data/pend-gdis-1960-2018-disasterlocations-gdb/pend-gdis-1960-2018-disasterlocations.gdb", 
            columns=['disasterno', 'disastertype', 'geometry'],
            ignore_geometry=False,
            rows=num_rows
        )

        gdis_df.drop(gdis_df[gdis_df['disastertype'].isin(['volcanic activity', 'earthquake'])].index, inplace=True)

        return gdis_df
    
    def get_intersecting_disasters_df(self, mask: Polygon):
        gdis_df = gpd.read_file(
            filename="data/pend-gdis-1960-2018-disasterlocations-gdb/pend-gdis-1960-2018-disasterlocations.gdb", 
            columns=['disasterno', 'disastertype', 'geometry'],
            ignore_geometry=False,
            mask=mask
        )

        gdis_df.drop(gdis_df[gdis_df['disastertype'].isin(['volcanic activity', 'earthquake'])].index, inplace=True)

        return gdis_df

    spacetime_query = "SELECT disastertype, total_deaths, num_injured, damage_cost, geometry FROM disasters WHERE start_date <= ? AND end_date >= ?"

    def query_spatiotemporal_point(self, longitude: float, latitude: float, timestamp: str):
        """
        Args:
            longitude (float): the longitude of chosen point to query
            latitude (float): the latitude of chosen point to query
            timestamp (str): the time of when to query; formatted like 'YYYYMMDD'
        """

        location_pin = Point(longitude, latitude) # chosen by user
        threshold = 30 # TODO: is arbitrary, so test

        with sqlite3.connect(self.db_file_path) as conn:
            cur = conn.cursor()
            cur.execute(
                self.spacetime_query, 
                (timestamp, timestamp)
            )
            results = cur.fetchall()
            cur.close()

        events = []
        for entry in results:
            geom = wkt.loads(entry[4])
            dist = location_pin.distance(geom.centroid)
            if dist < threshold and self.is_inside(location_pin, geom):
                # disasterno, total_deaths, num_injured, damage_cost, start_date, end_date, disastertype, geometry
                # disastertype, total_deaths, num_injured, damage_cost
                events.append([entry[0], entry[1], entry[2], entry[3]])
                break # TODO: perhaps disable later to account for multiple weather events at the same time in the same location

        if not events:
            return [['nothing', 0, 0, 0]]

        return events

    def is_inside(self, point, geometry):
        gdf = gpd.GeoSeries([geometry])
        return gdf.contains(point)[0]
    
    def clear_all(self):
        """
        Deletes all entries from the table.
        """
        with sqlite3.connect(self.db_file_path) as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM disasters")
            conn.commit()
            cur.close()

    def check_table_exists(self):
        # SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';
        with sqlite3.connect(self.db_file_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='disasters';")
            result = cur.fetchone()
            cur.close()
        return result is not None