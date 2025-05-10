from typing import List
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely import is_valid
import h3
from rtree import index
from src.utility import Timer

from src.configuration.config import Config
from src.datamodels import LandQueryModel

import logging
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
        pass
    
    def generate_world_polygons(self):
        h3_indexes = []

        for h3_index in h3.get_res0_cells():
            h3_indexes.extend(h3.cell_to_children(h=h3_index, res=Config.hexagon_resolution))

        logging.info(f"Total hexagons: {len(h3_indexes)}")

        # polygons = [make_valid(Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(h)])) for h in h3_indexes]
        polygons = [Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(h)]) for h in h3_indexes]

        polygons = [polygon for polygon in polygons if is_valid(polygon) and polygon.area < 55]

        int_h3_indexes = [h3.str_to_int(h3_index) for h3_index in h3_indexes]

        return polygons, int_h3_indexes
    
    def extract_land_hexagons(self, hexagons: List[BaseGeometry], hexagon_ids: List[int]):
        logging.info("Making rtree")
        # Making rtree

        gdf = LandQueryModel.get_table()

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
                if has_intersection(hexagon):
                    land_hexagons.append(hexagon)
                    land_hexagon_ids.append(hexagon_id)
        
        return land_hexagons, land_hexagon_ids
    
    def get_result(self):
        # returns h3 indexes as integers, then centroids of the valid cells
        # Generate hexagon polygons across the world
        logging.info("Creating hexagons...")

        polygons, int_h3_indexes = self.generate_world_polygons()

        if Config.show_init_hexagons:
            LandQueryModel.show_geoms_on_world(polygons=polygons)

        polygons, int_h3_indexes = self.extract_land_hexagons(
            hexagons=polygons, 
            hexagon_ids=int_h3_indexes
        )

        if Config.show_post_hexagons:
            LandQueryModel.show_geoms_on_world(polygons=polygons)

        polygon_centroids = [polygon.centroid for polygon in polygons]

        # into (long, lat) tuples
        latlon_points = [(point.x, point.y) for point in polygon_centroids]

        return int_h3_indexes, latlon_points

    def get_group_id(self, longitude: float, latitude: float) -> int:
        group_id = h3.str_to_int(
            h3.latlng_to_cell(
                lat=latitude, 
                lng=longitude, 
                res=Config.hexagon_resolution
            )
        )
        return group_id
    
    def get_coordinate_from_id(self, int_h3_index: int):
        h3_index = h3.int_to_str(int_h3_index)

        hexagon = Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(h3_index)])

        point = hexagon.centroid

        return (point.x, point.y)
    
    def save_ids(self, h3_indexes: List[int]):
        import pickle

        with open('db/h3_idxs.pkl', 'wb') as f:
            pickle.dump(h3_indexes, f)

    def load_ids(self):
        import pickle

        with open('db/h3_idxs.pkl', 'rb') as f:
            loaded_list = pickle.load(f)

        return loaded_list



        



