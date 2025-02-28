from typing import overload
import geopandas as gpd
from shapely.geometry import Point, MultiPolygon, Polygon
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)

class LandQueryService:
    land_filename = 'db/ne_10m_land'
    def __init__(self):
        self.gdf = gpd.read_file(
            filename=LandQueryService.land_filename, 
            columns=["featurecla", "geometry"]
        )

        logging.info(self.gdf.columns)
        logging.info(self.gdf)

        self.gdf.dropna(inplace=True)

        logging.info(self.gdf)

        self.gdf.plot()
        # plt.show()


    # def is_on_land(self, longitude: float, latitude: float):
    #     point = Point(longitude, latitude)

    #     return self.gdf.geometry.contains(point).any()
    
    def is_on_land(self, geopoint: tuple[float, float]):
        # geopoint is (longitude, latitude)
        point = Point(*(geopoint))

        return self.gdf.geometry.contains(point).any()
    
    def is_in_water(self, geopoint: tuple[float, float]):
        return not self.is_on_land(geopoint=geopoint)
    


    # https://stackoverflow.com/questions/49558464/shrink-polygon-using-corner-coordinates