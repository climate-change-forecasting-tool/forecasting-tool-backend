import geopandas as gpd

# Read a GeoPackage file
gdf = gpd.read_file(
    filename="data/pend-gdis-1960-2018-disasterlocations-gpkg/pend-gdis-1960-2018-disasterlocations.gpkg"
) # 39953 entries

# Perform operations on the GeoDataFrame
# For example, print the first few rows
# print(gdf.head())
# print(gdf.columns)
# print(gdf[0].geometry)
print(gdf.shape[0])

# Write a GeoDataFrame to a GeoPackage file
# gdf.to_file("data/pend-gdis-1960-2018-disasterlocations-gpkg/pend-gdis-1960-2018-disasterlocations.gpkg", driver="GPKG")