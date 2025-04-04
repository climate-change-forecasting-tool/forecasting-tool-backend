from datetime import datetime
import os
import geopandas as gpd
from shapely.geometry import Point, MultiPolygon, Polygon
from shapely.ops import unary_union, triangulate
import numpy as np
import matplotlib.pyplot as plt
import logging
import h3
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from rtree import index

from src.configuration.config import Config
from src.services.disaster_db_service import DisasterDBService

logging.basicConfig(level=logging.INFO)


"""
process:
1. Generate carpet of polygons (h3 hexagons)
2. Load all disaster geometries
3. Intersect carpet of polygons with ALL disaster geometries
    a. Remove polygon from carpet of polygons if it doesn't intersect (?)
4. Generate random point inside each polygon
5. Create new database with points and disaster data
"""

DB_FILE_PATH = "db/disasters.parquet"

class PointGenerator:
    # We will save the data produced by this and load it afterward; will just be a db, not gdb
    def __init__(self, resolution: int = 2): # lower resolution = larger area
        self.resolution = resolution

        self.schema = pa.schema([
            ("timestamp", pa.date64()),
            ("longitude", pa.float64()),
            ("latitude", pa.float64()),
            ("disastertype", pa.string()),
            ("total_deaths", pa.int64()),
            ("num_injured", pa.int64()),
            ("damage_cost", pa.int64()),
        ])

        self.create_table()

    def create_table(self):
        if os.path.exists(DB_FILE_PATH):
            if Config.recreate_disaster_database:
                logging.info("Clearing Disaster Parquet file and recreating now...")
                self.clear_dataset()
            else:

                logging.info("Skipping Disaster Parquet file creation.")
                return
        else:
            logging.info("Disaster Parquet file does not exist. Creating now...")

        # Generate hexagon polygons across the world
        logging.info("Creating hexagons...")
        h3_indexes = h3.polygon_to_cells(
            h3.LatLngPoly([(-180, -90), (180, -90), (180, 90), (-180, 90), (-180, -90)]),
            self.resolution
        )

        polygons = [Polygon(h3.cell_to_boundary(h)) for h in h3_indexes]

        disaster_serv = DisasterDBService()

        gdf = disaster_serv.controller.get_table()

        # Make a union of all of the disaster regions
        logging.info("Creating union of disasters...")
        combined_disasters = unary_union(gdf.geometry)

        # Reduce the number of polygons to only polygons that intersect with disaster regions
        logging.info("Removing hexagons that don't have disasters...")
        disaster_polygons = [h for h in polygons if h.intersects(combined_disasters)]

        # Intersect the disaster and polygon regions
        logging.info("Intersecting hexagons with disasters...")
        disaster_intersections = [h.intersection(combined_disasters) for h in disaster_polygons]

        # Generate a random point in the intersected region
        logging.info("Generating points within hexagons...")
        carpet_points = [self.random_point_in_polygon(p) for p in disaster_intersections]

        # TODO: use rtree spatial indexing to boost processing speed of geometry intersections
        
        # if using rtree, we don't necessarily need to have an exact intersection of polygon with hexagon & one single disaster,
        # then choose random point in that intersection

        # # Making rtree
        # idx = index.Index()

        # for idx_gdf, geometry in gdf.iterrows():
        #     # Insert the bounding box of each geometry (geometry.bounds is a tuple of (minx, miny, maxx, maxy))
        #     idx.insert(idx_gdf, geometry.geometry.bounds)

        parquet_writer = pq.ParquetWriter(where=DB_FILE_PATH, schema=self.schema)

        # saving longitude, latitude, dates, disastertype, num deaths, num injuries, property damage cost
        logging.info("Saving values...")
        for point in carpet_points:
            filtered_gdf = gdf[gdf.contains(point)]

            logging.info(filtered_gdf.columns)

            logging.info(filtered_gdf)
            # exit(0)

            for row in filtered_gdf.itertuples():
                dates = pd.date_range(start=row.start_date, end=row.end_date, freq="D", inclusive="both")

                temp_df = pd.DataFrame(
                    {
                        "timestamp": dates,
                        "longitude": [row.longitude] * len(dates),
                        "latitude": [row.latitude] * len(dates),
                        "disastertype": [row.disastertype] * len(dates),
                        "total_deaths": [row.total_deaths] * len(dates),
                        "num_injured": [row.num_injured] * len(dates),
                        "damage_cost": [row.damage_cost] * len(dates),
                    }
                )

                parquet_writer.write_table(table=temp_df)

        parquet_writer.close()

        logging.info("Disaster point generation process complete")

    def random_point_in_polygon(self, polygon: Polygon) -> Point:
        # Triangulate the polygon
        triangles = triangulate(polygon)
        
        # Compute areas and cumulative distribution
        areas = np.array([t.area for t in triangles])
        cumulative_areas = np.cumsum(areas)
        
        # Pick a triangle based on area weights
        # TODO: set a seed

        r = np.random.uniform(0, cumulative_areas[-1])
        chosen_triangle = triangles[np.searchsorted(cumulative_areas, r)]

        # Generate a random point inside the chosen triangle using barycentric coordinates
        p1, p2, p3 = chosen_triangle.exterior.coords[:3]
        r1, r2 = np.sqrt(np.random.uniform()), np.random.uniform()
        x = (1 - r1) * p1[0] + (r1 * (1 - r2)) * p2[0] + (r1 * r2) * p3[0]
        y = (1 - r1) * p1[1] + (r1 * (1 - r2)) * p2[1] + (r1 * r2) * p3[1]

        return Point(x, y)
    
    def get_all_points(self):
        pq_table = pd.read_parquet(
            path=DB_FILE_PATH, 
            engine='pyarrow',
            columns=['longitude', 'latitude']
        ).drop_duplicates(keep='first')

        points = list(pq_table.itertuples(index=False, name=None))

        return points
    
    def get_data_for_point(self, longitude: float, latitude: float):
        pq_table = pd.read_parquet(
            path=DB_FILE_PATH, 
            engine='pyarrow',
        )

        dates = pd.date_range(start=Config.start_date, end=Config.end_date, freq="D", inclusive="both")

        pq_table = pq_table[(pq_table['longitude'] == longitude) & (pq_table['latitude'] == latitude)]

        pq_table.drop(['longitude', 'latitude'], axis=1, inplace=True)

        dates = set(pq_table['date']).difference(set(dates))

        nodisaster_table = pd.DataFrame({
            "timestamp": dates,
            "disastertype": ["none"] * len(dates),
            "total_deaths": [0] * len(dates),
            "num_injured": [0] * len(dates),
            "damage_cost": [0] * len(dates),
        })

        merged = pd.merge(pq_table, nodisaster_table, on="timestamp", how="inner")

        merged.set_index('timestamp', inplace=True)

        merged.sort_index(inplace=True)

        return merged

    def clear_dataset(self):
        # Create an empty table
        empty_table = pa.Table.from_batches([], schema=self.schema)

        # Write empty Parquet file
        pq.write_table(empty_table, DB_FILE_PATH)



        



