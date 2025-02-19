from src.controllers.disaster_db_controller import DisasterDBController
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import itertools

import logging
logging.basicConfig(level=logging.INFO)

DB_FILE_PATH = "data/disasters.db"

class DisasterDBService:
    def __init__(self, db_file_path: str = DB_FILE_PATH):
        self.controller = DisasterDBController(db_file_path=db_file_path)

    def generate_biased_points(self, long_side: int = 100, lat_side: int = 50):
        """
        Generates ``long_side`` * ``lat_side`` geographic points that are biased toward the equator around the world

        Args:
            long_side (int): number of vertical divisions across the world. 
                Includes the Prime Meridian if this is an even number.
            lat_side (int): number of horizontal divisions across the world. Ensure this is greater than 2.
                Best for this to be an odd number so that it has points on the equator.

        """
        
        # Generate longitude uniformly between -180 and 180, but exclude 180 from the bounds, b/c it is the same as -180
        longitudes = np.linspace(start=-180, stop=180, num=long_side, endpoint=False)

        logging.info(longitudes)
        
        # Generate latitude uniformly biased toward equator between -90 and 90
        tan_bound = np.pi / 4.
        latitudes = np.linspace(start=-tan_bound, stop=tan_bound, num=lat_side)

        latitudes = np.tan(latitudes) * 90.

        logging.info(latitudes)

        # exclude poles from the cartesian product to exclude overlap
        points = list(itertools.product(longitudes, latitudes[1:-1]))

        # add poles
        points.append((0, -90.0))
        points.append((0, 90.0))
        
        return points

    # latitude [-90, 90) and longitude [-180, 180)
    def world_query(self, start_date: str, end_date: str):
        """
        Args:
            start_date (str): formatted like 'YYYYMMDD'
            end_date (str): formatted like 'YYYYMMDD'
        """

        result_df = pd.DataFrame(columns=['disastertype', 'total_deaths', 'num_injured', 'damage_cost'])

        points = self.generate_biased_points(long_side=16, lat_side=11)

        logging.info(points)
        
        end_datetime = datetime.strptime(end_date, "%Y%m%d")

        for (longitude, latitude) in points:
            current_datetime = datetime.strptime(start_date, "%Y%m%d")
            while current_datetime <= end_datetime:
                result_df.loc[len(result_df)] = \
                    self.controller.query_spatiotemporal_point(
                        longitude=longitude, 
                        latitude=latitude, 
                        timestamp=current_datetime.strftime("%Y%m%d")
                    )[0] # fix to account for more
                current_datetime += timedelta(days=1)

        return result_df




