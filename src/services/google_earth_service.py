# from datetime import datetime
# from typing import List
# import ee
# import pandas as pd

# from src.configuration.config import Config

# # only provides data from 2000-2021

# # Run `earthengine authenticate` in terminal

# class GoogleEarthService:
#     def __init__(self):
#         ee.Initialize(project=Config.project_name)

#     def get_population_data(self, points):
#         start_year = Config.start_date.year
#         end_year = Config.end_date.year

#         years = list(range(start_year, end_year+1))

#         # Create a multi-band image where each band is population for a specific year
#         bands = []

#         for year in years:
#             start_date = f"{year}-01-01"
#             if year == start_year:
#                 start_date = Config.start_date.strptime("%Y-%m-%d")
#             end_date = f"{year}-12-31"
#             if year == end_year:
#                 end_date = Config.end_date.strptime("%Y-%m-%d")
#             img = ee.ImageCollection("WorldPop/GP/100m/pop") \
#                 .filterDate(start_date, end_date) \
#                 .mosaic() \
#                 .rename(f"pop_{year}")
#             bands.append(img)

#         # Combine into one multi-band image
#         multi_band_image = ee.Image.cat(bands)

#         # Build FeatureCollection of buffered points
#         features = []
#         for pt in points:
#             geom = ee.Geometry.Point([pt[0], pt[1]]).buffer(Config.buffer_radius)
#             feature = ee.Feature(geom, {
#                 'lon': pt[0],
#                 'lat': pt[1]
#             })
#             features.append(feature)

#         fc = ee.FeatureCollection(features)

#         # Reduce all bands (years) at once
#         reduced = multi_band_image.reduceRegions(
#             collection=fc,
#             reducer=ee.Reducer.mean(),
#             scale=100,
#             maxPixels=1e9
#         ).getInfo()

#         pop_data = []

#         for feature in reduced['features']:
#             props = feature['properties']
#             longitude = props['lon']
#             latitude = props['lat']

#             for year in years:
#                 start_date = datetime(year, 1, 1)
#                 if year == start_year:
#                     start_date = Config.start_date
#                 end_date = datetime(year, 12, 31)
#                 if year == end_year:
#                     end_date = Config.end_date
#                 population = props.get(f"pop_{year}")
#                 year_as_daily = pd.date_range(start=start_date, end=end_date, freq="D", inclusive="both")
#                 pop_data.extend([(date, longitude, latitude, population) for date in year_as_daily])

#         population_df = pd.DataFrame(pop_data, columns=['timestamp', 'longitude', 'latitude', 'population'])

#         population_df.sort_values(['longitude', 'latitude', 'timestamp'], inplace=True)

#         return population_df
    
#     @staticmethod
#     def query_point_in_df(df, longitude: float, latitude: float, columns: List[str]):
#         point_df = df[(df['longitude'] == longitude) & (df['latitude'] == latitude)]

#         return point_df[columns]