import geopandas as gpd
import pandas as pd

from shapely.geometry import Point, MultiPolygon, Polygon

# 'id', 'country', 'iso3', 'gwno', 'geo_id', 'geolocation', 'level', 'adm1', 'adm2', 'adm3', 'location', 
# 'historical', 'hist_country', 'disastertype', 'disasterno', 'Shape_Length', 'Shape_Area'

gdis_df = gpd.read_file(
    filename="data/pend-gdis-1960-2018-disasterlocations-gdb/pend-gdis-1960-2018-disasterlocations.gdb", 
    columns=['disasterno', 'iso3', 'disastertype', 'geometry'],
    ignore_geometry=False
) # 39952 entries

print(gdis_df)

# filter out entries that have null iso3's
gdis_df = gdis_df.loc[gdis_df['iso3'].notnull()]

# add iso3 to disasterno to get the corresponding EM-DAT disaster number
gdis_df['disasterno'] += '-' + gdis_df['iso3']

# drop iso3
gdis_df.drop('iso3', axis=1, inplace=True)

gdis_df.drop_duplicates(keep='first', inplace=True) # 9780

print(gdis_df)

emdat_df = pd.read_excel(
    io='data/emdat_disasterdata.xlsx',
    usecols=['DisNo.', 'Start Year', 'Start Month', 'Start Day', 'End Year', 'End Month', 'End Day', 'Total Deaths', 'No. Injured', "Total Damage, Adjusted ('000 US$)"]
) # 26090 entries

# rename disaster name
emdat_df.rename(columns={'DisNo.': 'disasterno', 'Total Deaths': 'total_deaths', 'No. Injured': 'num_injured', "Total Damage, Adjusted ('000 US$)": 'damage_cost'}, inplace=True)

# filter out entries that have empty start dates or end dates
emdat_df = emdat_df[(emdat_df['Start Month'].notnull()) & (emdat_df['Start Day'].notnull()) & (emdat_df['End Month'].notnull()) & (emdat_df['End Day'].notnull())]

emdat_df[['Start Year', 'Start Month', 'Start Day', 'End Year', 'End Month', 'End Day']] = \
    emdat_df[['Start Year', 'Start Month', 'Start Day', 'End Year', 'End Month', 'End Day']].astype(int).astype(str)

emdat_df['start_date'] = emdat_df['Start Year'].str.zfill(4) + emdat_df['Start Month'].str.zfill(2) + emdat_df['Start Day'].str.zfill(2)
emdat_df['end_date'] = emdat_df['End Year'].str.zfill(4) + emdat_df['End Month'].str.zfill(2) + emdat_df['End Day'].str.zfill(2)

emdat_df.drop(['Start Year', 'Start Month', 'Start Day', 'End Year', 'End Month', 'End Day'], axis=1, inplace=True) # 22105

print(emdat_df)

combined_df = pd.merge(left=emdat_df, right=gdis_df, on='disasterno', how='inner') # 7326

combined_df.set_index('disasterno', inplace=True)

print(combined_df)

from shapely import wkb, wkt
import sqlite3

# sqlite3 storage of data

def query_spatiotemporal_point(longitude: float, latitude: float, timestamp: str):
    """
    Args:
        longitude (float): the longitude of chosen point to query
        latitude (float): the latitude of chosen point to query
        timestamp (str): the time of when to query; formatted like 'YYYYMMDD'
    """

    location_pin = Point(longitude, latitude) # chosen by user

    # some way to fetch the geometry
    date_entries = combined_df.loc[combined_df['start_date'].ge(timestamp) & combined_df['end_date'].le(timestamp)]

    matching_entries = pd.DataFrame(columns=combined_df.columns)
    for row in date_entries.itertuples():
        if is_inside(location_pin, row.geometry):
            matching_entries.loc[len(matching_entries)] = row

    return matching_entries

def is_inside(point, geometry):
    gdf = gpd.GeoSeries([geometry])
    return gdf.contains(point)[0]
