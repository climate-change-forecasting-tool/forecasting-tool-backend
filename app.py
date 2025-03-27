from multiprocessing import freeze_support

# latitude [-90, 90] and longitude [-180, 180)

if __name__ == '__main__':
    import src

    freeze_support()



"""

Temporal extent:
gdis: (1960-01-01)-(2018-12-31)
NASA climate data: (1980-12-31)-(?)


Spatial extent:
gdis: 90 N, -58 S, 180 E, -180 W



Data processing
1. Generate points around the globe to query (including a bounding box radius)
2. Use some timespan: start date to end date
3. For each point at some time, retrieve:
    a. timestamp
    b. longitude
    c. latitude
    d. elevation                    NASA POWER API
    e. disastertype                 EM-DAT / GDIS
    f. number of deaths             EM-DAT
    g. number of injuries           EM-DAT
    h. property damage cost         EM-DAT
    i. ***weather data
        - temperature               NASA POWER API
        - precipitation             NASA POWER API
        - humidity                  NASA POWER API
        - air pressure              NASA POWER API
        - wind speed                NASA POWER API
        - wind direction            NASA POWER API
        - snow cover
    j. ***climate-change data (pollutants)
        - 
    k. ***urban data
        - population density
        - 
4. Save to dataset



For processing data either:
    - Quadtrees for disaster geometry centroids
    - Keeping all disaster geometries in memory
        - Query disaster dataset for dates 
    
- Use population data from CARTO

Number of points for a given tile side length:
    y = 0.3*(40000/x)^2
    where x is distance in kilometers

"""