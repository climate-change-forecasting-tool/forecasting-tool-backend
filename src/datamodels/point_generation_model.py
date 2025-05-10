import os
from typing import List
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely import is_valid
import numpy as np
import logging
import h3
from rtree import index
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from src.utility import Timer, flatten_list

from src.configuration.config import Config
from .climate_model import GRIB_Controller

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

class SummaryDataset:
    # We will save the data produced by this and load it afterward; will just be a db, not gdb
    def __init__(self): # lower resolution = larger area
        self.resolution = Config.hexagon_resolution

        self.climate_params = []

        # print(flatten_list(Config.cams_files_and_vars.items()))

        for climate_var_name in flatten_list(Config.cams_files_and_vars.values()):
            self.climate_params.append((climate_var_name + '_min', pa.float64()))
            self.climate_params.append((climate_var_name + '_mean', pa.float64()))
            self.climate_params.append((climate_var_name + '_max', pa.float64()))


        self.schema = pa.schema([
            ("timestamp", pa.uint64()),
            ("group_id", pa.int64()),
            ("longitude", pa.float64()),
            ("latitude", pa.float64()),
        ] + self.climate_params)

        self.lqm = LandQueryModel()

        self.create_table()

    def create_table(self):
        if os.path.exists(Config.summary_dataset_filepath):
            if Config.recreate_summary_dataset:
                logging.info("Clearing Summary Dataset Parquet file and recreating now...")
                self.clear_dataset()
            else:
                logging.info("Skipping Summary Dataset Parquet file creation.")
                return
        else:
            logging.info("Summary Dataset Parquet file does not exist. Creating now...")
            self.clear_dataset()

        # Generate hexagon polygons across the world
        logging.info("Creating hexagons...")

        polygons, int_h3_indexes = self.generate_world_polygons()

        if Config.show_init_hexagons:
            self.lqm.show_geoms_on_world(polygons=polygons)

        # IF ONLY WANTING LAND DATA, DO RTREE INTERSECTION STUFF HERE

        polygons, int_h3_indexes = self.extract_land_hexagons(
            hexagons=polygons, 
            hexagon_ids=int_h3_indexes
        )

        if Config.show_post_hexagons:
            self.lqm.show_geoms_on_world(polygons=polygons)

        polygon_centroids = [polygon.centroid for polygon in polygons]

        grib_controller = GRIB_Controller()

        parquet_writer = pq.ParquetWriter(where=Config.summary_dataset_filepath, schema=self.schema)

        # saving longitude, latitude, dates, disastertype, num deaths, num injured, property damage cost
        logging.info("Saving values...")
        with Timer("Saving values"):
            for idx, (h3_index, point) in enumerate(zip(int_h3_indexes, polygon_centroids)):
                with Timer(f"lon={point.x}, lat: {point.y}; group_id: {h3_index}; ({idx+1}/{len(int_h3_indexes)})"):
                    climate_df = grib_controller.get_point_data(
                        longitude=point.x,
                        latitude=point.y
                    )

                    # print("Climate df:")
                    # print(climate_df)

                    # print(climate_df.index.to_series())

                    dates = (climate_df.index.to_series() - Config.start_date).dt.days.astype(int)

                    # print(dates)

                    entry_dict = dict({
                        "timestamp": pa.array(dates, type=pa.uint64()),
                        "group_id": pa.array([h3_index] * len(dates), type=pa.int64()),
                        "longitude": pa.array([point.x] * len(dates), type=pa.float64()),
                        "latitude": pa.array([point.y] * len(dates), type=pa.float64()),
                    })

                    for climate_param, pa_type in self.climate_params:
                        entry_dict.update({climate_param: pa.array(climate_df[climate_param], type=pa_type)})

                    entry = pa.table(entry_dict)

                    parquet_writer.write_table(table=entry)

        parquet_writer.close()

        logging.info("Summary dataset created")
    
    def generate_world_polygons(self):
        h3_indexes = []

        for h3_index in h3.get_res0_cells():
            h3_indexes.extend(h3.cell_to_children(h=h3_index, res=self.resolution))

        logging.info(f"Total hexagons: {len(h3_indexes)}")

        # polygons = [make_valid(Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(h)])) for h in h3_indexes]
        polygons = [Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(h)]) for h in h3_indexes]

        polygons = [polygon for polygon in polygons if is_valid(polygon) and polygon.area < 55]

        int_h3_indexes = [h3.str_to_int(h3_index) for h3_index in h3_indexes]

        return polygons, int_h3_indexes
    
    def extract_land_hexagons(self, hexagons: List[BaseGeometry], hexagon_ids: List[int]):
        logging.info("Making rtree")
        # Making rtree

        gdf = self.lqm.get_table()

        with Timer("Rtree") as t:
            idx = index.Index()

            gdf['id'] = range(1, len(gdf) + 1)
            gdf.set_index('id', inplace=True)

            for row in gdf.itertuples(index=True):
                # Insert the bounding box of each geometry (geometry.bounds is a tuple of (minx, miny, maxx, maxy))
                idx.insert(int(row.Index), row.geometry.bounds)

        logging.info("Making intersections")
        with Timer("Intersections") as t:
            def has_intersection(polygon: Polygon):
                possible_matches = list(idx.intersection(polygon.bounds))  # Get candidate polygons
                for match_id in possible_matches:
                    candidate_polygon = gdf.loc[match_id, "geometry"]
                    intersection = polygon.intersection(candidate_polygon)
                    if not intersection.is_empty:
                        return True
                return False
            
            land_hexagons = []
            land_hexagon_ids = []
            for hexagon, hexagon_id in zip(hexagons, hexagon_ids):
                # is_intersecting = has_intersection(hexagon)
                # if is_intersecting:
                if has_intersection(hexagon):
                    land_hexagons.append(hexagon)
                    land_hexagon_ids.append(hexagon_id)
        
        return land_hexagons, land_hexagon_ids

    def clear_dataset(self):
        # Create an empty table
        empty_table = pa.Table.from_batches([], schema=self.schema)

        # Write empty Parquet file
        pq.write_table(empty_table, Config.summary_dataset_filepath)

    def get_group_id(self, longitude: float, latitude: float) -> int:
        group_id = h3.str_to_int(
            h3.latlng_to_cell(
                lat=latitude, 
                lng=longitude, 
                res=Config.hexagon_resolution
            )
        )
        return group_id



        



