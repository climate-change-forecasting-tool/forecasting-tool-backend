from typing import overload
import geopandas as gpd
from shapely.geometry import Point, MultiPolygon, Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import logging
import numpy as np
import math

logging.basicConfig(level=logging.INFO)

class LandQueryService:
    land_filename = 'db/ne_10m_land'
    def __init__(self):
        self.gdf = gpd.read_file(
            filename=LandQueryService.land_filename,
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

        # logging.info(self.gdf)


    """
    1. Get contours of land by using the buffer & some fixed step_distance

    2. Repeat until the retrieved polygon is empty

    3. Take points by 
    """

    # def get_evenly_spaced_points(self, polygon: Polygon, num_points: int):
    #     """Returns 'num_points' evenly spaced points along the boundary of a polygon"""
    #     exterior = polygon.exterior  # Get the outer boundary
    #     total_length = exterior.length  # Perimeter of the polygon
    #     distances = np.linspace(0, total_length, num_points, endpoint=False)  # Divide perimeter into segments
    #     points = [exterior.interpolate(d) for d in distances]  # Get points at distances
    #     return points
    
    def get_evenly_spaced_points_bd(self, polygon: Polygon, step_distance: float):
        """Returns evenly spaced points along the boundary of a polygon by distance"""
        exterior = polygon.exterior  # Get the outer boundary
        total_length = exterior.length  # Perimeter of the polygon
        num_points = math.ceil(total_length / step_distance)
        distances = np.linspace(0, total_length, num_points, endpoint=False)  # Divide perimeter into segments
        def point_to_tuple(p: Point):
            return (p.x, p.y)
        points = [point_to_tuple(exterior.interpolate(d)) for d in distances]  # Get points at distances
        return points
    
    def get_contours(self, polygon: Polygon, step_distance: float):
        contours = [] # list of polygons

        # if we want to keep the first polygon
        contours.append(polygon)

        def rec_contours(polygon: Polygon):
            new_polygon = polygon.buffer(-step_distance)

            if new_polygon.is_empty:
                return

            if(isinstance(new_polygon, MultiPolygon)):
                contours.extend(new_polygon.geoms)
                for broken_polygon in new_polygon.geoms:
                    rec_contours(polygon=broken_polygon)
            else:
                contours.append(new_polygon)
                rec_contours(polygon=new_polygon)
            return

        rec_contours(polygon=polygon)
        
        return contours
    
    def get_contour_points(self, poly: Polygon, inward_step: float, perimeter_step: float):
        points = []

        points.extend(self.get_evenly_spaced_points_bd(poly, perimeter_step))

        def rec_contour_points(polygon: Polygon):
            new_polygon = polygon.buffer(-inward_step)

            if new_polygon.is_empty:
                return

            if(isinstance(new_polygon, MultiPolygon)):
                for broken_polygon in new_polygon.geoms:
                    points.extend(self.get_evenly_spaced_points_bd(broken_polygon, perimeter_step))
                    rec_contour_points(polygon=broken_polygon)
            else:
                points.extend(self.get_evenly_spaced_points_bd(new_polygon, perimeter_step))
                rec_contour_points(polygon=new_polygon)
            return

        rec_contour_points(polygon=poly)
        return points
    
    def get_world_points(self, inward_step: float, perimeter_step: float):
        all_points = []

        # TODO: simply do not consider points that are below -60 latitude
        for n_polygon in self.gdf.geometry:
            if isinstance(n_polygon, MultiPolygon):
                for polygon in n_polygon.geoms:
                    new_points = self.get_contour_points(poly=polygon, inward_step=inward_step, perimeter_step=perimeter_step)
                    all_points.extend(new_points)
            else:
                new_points = self.get_contour_points(poly=n_polygon, inward_step=inward_step, perimeter_step=perimeter_step)
                all_points.extend(new_points)

        # we will only take points that are above -57.5 latitude
        def filter_out_antarctic(p):
            return p[1] > -57.5

        all_points = list(filter(filter_out_antarctic, all_points))

        """
        TODO: create polygons that will filter out portions of the points within certain regions,
        i.e. Greenland, Northern Canada
        """

        return all_points

    def get_world_points_v2(self, long_step: float, lat_step: float):
        """
        basically generates boxes across the world and attempts to place a point on land somewhere in the box
        """
        pass

    def dostuff(self):
        import matplotlib.pyplot as plt

        lqs = LandQueryService()

        # plt.ion()

        fig, ax = plt.subplots(figsize=(10, 8))

        # logging.info(adg)

        # disaster_geom_df.geometry.plot()

        lqs.gdf.geometry.plot(ax=ax, color='white', edgecolor='black', alpha=1.0, label="Dataset 1")

        # plt.pause(10.0)

        from shapely.geometry import Point, MultiPolygon, Polygon

        # from src.controllers.disaster_db_controller import DisasterDBController

        # ddbs = DisasterDBController()

        # disaster_geomdf = ddbs.get_geometry_df()
        # logging.info("Done retrieving disaster geometries!")

        # centroids = []

        # # 'landslide', 'flood', 'volcanic activity', 'earthquake', 'mass movement (dry)', 'extreme temperature ', 'storm', 'drought'
        # disaster_to_color = {
        #     'landslide': 'blue', 
        #     'flood': 'darkviolet', 
        #     'mass movement (dry)': 'yellow', 
        #     'extreme temperature ': 'red', 
        #     'storm': 'darkorange', 
        #     'drought': 'green'
        # }
        # colors = [disaster_to_color[disaster] for disaster in disaster_geomdf['disastertype']]
        # logging.info("Done coloring disasters!")

        # disaster_geomdf.geometry.plot(ax=ax, color=colors, edgecolor='black', alpha=0.1, label="Dataset 2")
        # logging.info("Done plotting disasters")

        ###############################################################
        inward_step = 3.0
        perimeter_step=6.0
        world_points = lqs.get_world_points(inward_step=inward_step, perimeter_step=perimeter_step)

        decimate_perc = 0.05
        decimate_polygon = Polygon([(-10.4, 89.6), (-6.2, 84.0), (-21.8, 68.1), (-43.9, 59.0), (-65.2, 61.3), (-79.7, 63.6), (-94.4, 60.9), (-139.7, 60.5), (-141.1, 90.9)])
        frequency = math.ceil(1. / decimate_perc)
        def inbounds_decimate(point_coords, idx: int):
            is_inside = decimate_polygon.contains(Point(point_coords))
            if is_inside:
                return idx % frequency == 0
            return True # keep all that are out of this bound
        
        undrawn_points = [point for idx, point in enumerate(world_points) if not inbounds_decimate(point, idx)]
        
        logging.info(len(world_points))

        world_points[:] = [point for idx, point in enumerate(world_points) if inbounds_decimate(point, idx)]

        logging.info(len(world_points))

        x, y = zip(*world_points)

        plt.scatter(x, y, c='red')

        plt.pause(5)

        non_x, non_y = zip(*undrawn_points)

        plt.scatter(non_x, non_y, c='blue')

        # point = Point(-101.56, 28.77)

        # mask = point.buffer(distance=3.0)

        # x, y = mask.exterior.xy

        # ax.fill(x, y, color='red', alpha=1.0, label="Dataset 2")

        # plt.plot(ax=ax, color='red', edgecolor='black', alpha=1.0, label="Dataset 2")

        # disaster_query_df = ddbs.get_intersecting_disasters_df(mask=mask)

        # disaster_query_df.geometry.plot(ax=ax, color='green', edgecolor='black', alpha=1.0, label="Dataset 3")

        # for n_polygon in lqs.gdf.geometry:
        #     if isinstance(n_polygon, MultiPolygon):
        #         logging.info("checking multipolygon")
        #         for polygon in n_polygon.geoms:
        #             new_points = lqs.get_contour_points(poly=polygon, inward_step=inward_step, perimeter_step=perimeter_step)
        #             logging.info(len(new_points))
        #             new_x, new_y = zip(*new_points)
        #             plt.scatter(new_x, new_y, c='red')
        #             all_points.extend(new_points)
        #             # plt.pause(0.1)
        #     else:
        #         logging.info("checking polygon")
        #         new_points = lqs.get_contour_points(poly=n_polygon, inward_step=inward_step, perimeter_step=perimeter_step)
        #         logging.info(len(new_points))
        #         new_x, new_y = zip(*new_points)
        #         plt.scatter(new_x, new_y, c='red')
        #         all_points.extend(new_points)
        #         # plt.pause(0.1)
                
        #             # plt.pause(2)
        #     # plt.pause(0.1)
        #     plt.draw()

        # x, y = zip(*all_points)

        # logging.info(len(all_points))

        # plt.scatter(x, y, c='red')

        plt.show()


        import time
        while True:
            time.sleep(1)

