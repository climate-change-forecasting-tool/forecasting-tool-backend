from typing import List
import geopandas as gpd
from shapely.geometry import Point, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import logging
from src.configuration.config import Config

logging.basicConfig(level=logging.INFO)

"""
process:
1. Generate carpet of polygons (h3 hexagons)
2. Load all disaster geometries
3. Intersect carpet of polygons with ALL disaster geometries
    a. Remove polygon from carpet of polygons if it doesn't intersect (?)
4. Generate random point inside each polygon
5. Create new database with points and disaster data

land masses are used for querying new points from frontend
* We will want to save a unary union, so earth is just one big multipolygon
* WE WILL NOT USE LOCATIONS THAT HAVE NO DISASTER DATA FOR MODEL
"""

class LandQueryModel:
    def __init__(self):
        self.gdf = gpd.read_file(
            filename=Config.landmass_filepath,
            # columns=["featurecla", "geometry"],
            bbox=(-180, 90, 180, -60)
        )

        # the row with the "N/A" featurecla is a ton of tiny islands
        # the row with "Null island" is just a buouy at the (0,0)

        # logging.info(self.gdf.columns)
        # logging.info(self.gdf)

        # We only want to keep the major land features
        self.gdf.drop(self.gdf[self.gdf['featurecla'] != 'Land'].index, inplace=True)

        # We don't want to process small, likely uninhabited islands
        # self.gdf.drop(self.gdf[self.gdf['scalerank'] >= 3.0].index, inplace=True)

        # # drop Antarctica from consideration, b/c no important disasters will occur there
        self.gdf.loc[0, "geometry"] = MultiPolygon(self.gdf.geometry[0].geoms[1:])

        # self.combined_landmass = self.gdf.geometry.unary_union

    
    def is_on_land(self, longitude: float, latitude: float):
        return self.gdf.contains(Point(longitude, latitude))
    
    def show_geoms_on_world(self, polygons: List[BaseGeometry]):
        fig, ax = plt.subplots(figsize=(10, 8))
        self.gdf.geometry.plot(ax=ax, color='white', edgecolor='black', alpha=1.0, label="Dataset 1")
        patches = []
        for geom in polygons:
            if isinstance(geom, MultiPolygon):
                parts = geom.geoms
            elif isinstance(geom, Polygon):
                parts = [geom]
            else:
                raise TypeError(f"Expected Polygon or MultiPolygon, got {type(geom)}")

            for part in parts:
                if not part.is_empty:
                    exterior = list(part.exterior.coords)
                    patch = MplPolygon(exterior, closed=True)
                    patches.append(patch)

        patch_collection = PatchCollection(patches, facecolor="red", edgecolor="black", alpha=0.2)
        ax.add_collection(patch_collection)
        plt.show()
