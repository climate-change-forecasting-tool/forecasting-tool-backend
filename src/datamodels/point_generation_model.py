import os
from typing import List
from shapely.geometry import Point, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import triangulate
from shapely import make_valid
import numpy as np
import matplotlib.pyplot as plt
import logging
import h3
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from rtree import index
from src.utility import Timer, get_nonunique_elements

from src.configuration.config import Config
from .disaster_model import DisasterModel

from src.datamodels import LandQueryModel

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

class PointGenerationModel:
    # We will save the data produced by this and load it afterward; will just be a db, not gdb
    def __init__(self): # lower resolution = larger area
        self.resolution = Config.hexagon_resolution

        self.schema = pa.schema([
            ("timestamp", pa.date64()),
            ("longitude", pa.float64()),
            ("latitude", pa.float64()),
            ("has_landslide", pa.bool_()),
            ("has_flood", pa.bool_()),
            ("has_dry_mass_movement", pa.bool_()),
            ("has_extreme_temperature", pa.bool_()),
            ("has_storm", pa.bool_()),
            ("has_drought", pa.bool_()),
            ("total_deaths", pa.int64()),
            ("num_injured", pa.int64()),
            ("damage_cost", pa.int64()),
        ])

        self.create_table()

    def create_table(self):
        if os.path.exists(Config.combined_disaster_data_filepath):
            if Config.recreate_disaster_database:
                logging.info("Clearing Disaster Parquet file and recreating now...")
                self.clear_dataset()
            else:
                logging.info("Skipping Disaster Parquet file creation.")
                return
        else:
            if os.path.exists(Config.summary_dataset_filepath) and not Config.recreate_disaster_database:
                logging.info("Skipping Disaster Parquet file creation.")
                return
            else:
                logging.info("Disaster Parquet file does not exist. Creating now...")
                self.clear_dataset()

        # Generate hexagon polygons across the world
        logging.info("Creating hexagons...")

        polygons = self.generate_world_polygons()

        if Config.show_hexagons:
            LandQueryModel().show_geoms_on_world(polygons=polygons)

        disaster_model = DisasterModel()

        with Timer("Fetching disaster data") as t:
            gdf = disaster_model.get_table()
        
        if Config.show_disasters:
            logging.info("Populating world map with disasters...")
            LandQueryModel().show_geoms_on_world(polygons=gdf.geometry)

        logging.info("Making rtree")
        # Making rtree
        with Timer("Rtree") as t:
            idx = index.Index()

            gdf['id'] = range(1, len(gdf) + 1)
            gdf.set_index('id', inplace=True)

            for row in gdf.itertuples(index=True):
                # Insert the bounding box of each geometry (geometry.bounds is a tuple of (minx, miny, maxx, maxy))
                idx.insert(int(row.Index), row.geometry.bounds)

        logging.info("Making intersections")
        with Timer("Intersections") as t:
            def get_intersection(polygon: Polygon):
                possible_matches = list(idx.intersection(polygon.bounds))  # Get candidate polygons
                for match_id in possible_matches:
                    candidate_polygon = gdf.loc[match_id, "geometry"]
                    intersection = polygon.intersection(candidate_polygon)
                    if not intersection.is_empty:
                        return intersection
                return None
            
            intersections = []
            for polygon in polygons:
                intersection = get_intersection(polygon)
                if intersection:
                    intersections.append(intersection)

        # Generate a random point in the intersected region
        logging.info("Generating points within hexagons...")
        with Timer("Random hexagon points") as t:
            carpet_points = [self.random_point_in_polygon(p) for p in intersections]

        parquet_writer = pq.ParquetWriter(where=Config.combined_disaster_data_filepath, schema=self.schema)

        # saving longitude, latitude, dates, disastertype, num deaths, num injured, property damage cost
        logging.info("Saving values...")
        with Timer("Saving values") as t:
            def get_intersection_ids(point: Point):
                possible_matches = list(idx.intersection(point.bounds))  # Get candidate polygons
                true_intersection_ids = []
                for match_id in possible_matches:
                    candidate_polygon = gdf.loc[match_id, "geometry"]
                    if point.intersects(candidate_polygon):
                        true_intersection_ids.append(match_id)
                return true_intersection_ids

            for i, point in enumerate(carpet_points):
                filtered_gdf = gdf.loc[get_intersection_ids(point)]

                for row in filtered_gdf.itertuples():
                    dates = pd.date_range(start=row.start_date, end=row.end_date, freq="W", inclusive="both")

                    entry = pa.table({
                        "timestamp": pa.array(dates, type=pa.date64()),
                        "longitude": pa.array([point.x] * len(dates), type=pa.float64()),
                        "latitude": pa.array([point.y] * len(dates), type=pa.float64()),
                        "has_landslide": pa.array([row.disastertype == 'landslide'] * len(dates), type=pa.bool_()),
                        "has_flood": pa.array([row.disastertype == 'flood'] * len(dates), type=pa.bool_()),
                        "has_dry_mass_movement": pa.array([row.disastertype == 'mass movement (dry)'] * len(dates), type=pa.bool_()),
                        "has_extreme_temperature": pa.array([row.disastertype == 'extreme temperature '] * len(dates), type=pa.bool_()),
                        "has_storm": pa.array([row.disastertype == 'storm'] * len(dates), type=pa.bool_()),
                        "has_drought": pa.array([row.disastertype == 'drought'] * len(dates), type=pa.bool_()),
                        "total_deaths": pa.array([row.total_deaths] * len(dates), type=pa.int64()),
                        "num_injured": pa.array([row.num_injured] * len(dates), type=pa.int64()),
                        "damage_cost": pa.array([row.damage_cost] * len(dates), type=pa.int64()),
                    })

                    parquet_writer.write_table(table=entry)

        parquet_writer.close()

        logging.info("Disaster point generation process complete")
    
    def generate_world_polygons(self):
        h3_indexes = []

        for h3_index in h3.get_res0_cells():
            h3_indexes.extend(h3.cell_to_children(h=h3_index, res=self.resolution))

        logging.info(f"Total hexagons: {len(h3_indexes)}")

        polygons = [make_valid(Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(h)])) for h in h3_indexes]

        return polygons

    def random_point_in_polygon(self, polygon: Polygon) -> Point:
        # Triangulate the polygon
        triangles = triangulate(polygon)
        
        # Compute areas and cumulative distribution
        areas = np.array([t.area for t in triangles])
        cumulative_areas = np.cumsum(areas)
        
        # Pick a triangle based on area weights
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
            path=Config.combined_disaster_data_filepath, 
            engine='pyarrow',
            columns=['longitude', 'latitude']
        ).drop_duplicates(keep='first')

        points = list(pq_table.itertuples(index=False, name=None))

        return points
    
    def get_data_for_point(self, longitude: float, latitude: float):
        pq_table: pd.DataFrame = pd.read_parquet(
            path=Config.combined_disaster_data_filepath, 
            engine='pyarrow',
        )

        pq_table['timestamp'] = pd.to_datetime(pq_table['timestamp'])

        pq_table = pq_table[(Config.start_date <= pq_table['timestamp']) & (pq_table['timestamp'] <= Config.end_date)]

        dates = pd.date_range(start=Config.start_date, end=Config.end_date, freq="W", inclusive="both")

        pq_table = pq_table[(pq_table['longitude'] == longitude) & (pq_table['latitude'] == latitude)]

        pq_table.drop(['longitude', 'latitude'], axis=1, inplace=True)

        #### TODO: need to drop duplicates while keeping the most data; 
        #### ex: look up rows 3587 and 3645 in emdat, they have same # deaths, but diff. no. affected

        # temporary
        # pq_table.drop_duplicates(subset='timestamp', keep='first', inplace=True)
        dup_timestamps = get_nonunique_elements(pq_table['timestamp'])

        for dup_timestamp in dup_timestamps:
            dup_entries = pq_table[pq_table['timestamp'] == dup_timestamp]

            has_landslide = np.any(dup_entries['has_landslide'])
            has_flood = np.any(dup_entries['has_flood'])
            has_dry_mass_movement = np.any(dup_entries['has_dry_mass_movement'])
            has_extreme_temperature = np.any(dup_entries['has_extreme_temperature'])
            has_storm = np.any(dup_entries['has_storm'])
            has_drought = np.any(dup_entries['has_drought'])
            total_deaths = np.sum(dup_entries['total_deaths'])
            num_injured = np.sum(dup_entries['num_injured'])
            damage_cost = np.sum(dup_entries['damage_cost'])

            # delete all records with a matching duplicate timestamp
            pq_table = pq_table[pq_table['timestamp'] != dup_timestamp]

            pq_table.loc[len(pq_table)] = [
                dup_timestamp, 
                has_landslide,
                has_flood,
                has_dry_mass_movement,
                has_extreme_temperature,
                has_storm,
                has_drought,
                total_deaths,
                num_injured,
                damage_cost
            ]

        existing_dates = pq_table['timestamp']

        missing_dates = list(np.setdiff1d(pd.to_datetime(dates), existing_dates))
        
        nodisaster_table = pd.DataFrame({
            "timestamp": missing_dates,
            "has_landslide": pa.array([False] * len(missing_dates), type=pa.bool_()),
            "has_flood": pa.array([False] * len(missing_dates), type=pa.bool_()),
            "has_dry_mass_movement": pa.array([False] * len(missing_dates), type=pa.bool_()),
            "has_extreme_temperature": pa.array([False] * len(missing_dates), type=pa.bool_()),
            "has_storm": pa.array([False] * len(missing_dates), type=pa.bool_()),
            "has_drought": pa.array([False] * len(missing_dates), type=pa.bool_()),
            "total_deaths": [0] * len(missing_dates),
            "num_injured": [0] * len(missing_dates),
            "damage_cost": [0] * len(missing_dates),
        })

        nodisaster_table['timestamp'] = pd.to_datetime(nodisaster_table['timestamp'])

        # merge tables
        merged = pd.concat([pq_table, nodisaster_table], join="inner")

        merged.set_index('timestamp', inplace=True)
        merged.sort_index(inplace=True)

        return merged

    def clear_dataset(self):
        # Create an empty table
        empty_table = pa.Table.from_batches([], schema=self.schema)

        # Write empty Parquet file
        pq.write_table(empty_table, Config.combined_disaster_data_filepath)

    def get_group_id(self, longitude: float, latitude: float) -> int:
        group_id = h3.str_to_int(
            h3.latlng_to_cell(
                lat=latitude, 
                lng=longitude, 
                res=Config.hexagon_resolution
            )
        )
        return group_id



        



