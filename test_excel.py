
import sqlite3
import json
import os
from typing import List
import logging
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, MultiPolygon, Polygon
from shapely import wkb, wkt
import time
import numpy as np

emdat_df = pd.read_excel(
    io='data/emdat_disasterdata.xlsx',
    usecols=['DisNo.', 'Start Year', 'Start Month', 'Start Day', 'End Year', 'End Month', 'End Day', 'Total Deaths', 'No. Injured', "Total Damage, Adjusted ('000 US$)"]
) # 26090 entries

arr1 = np.nan_to_num(x=emdat_df['Total Deaths'].astype(np.string_).astype(np.float64), nan=0.0).astype(np.int64)

arr2 = np.nan_to_num(x=emdat_df['No. Injured'].astype(np.string_).astype(np.float64), nan=0.0).astype(np.int64)

print("done")