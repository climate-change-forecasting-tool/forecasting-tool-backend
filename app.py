from multiprocessing import freeze_support

# latitude [-90, 90] and longitude [-180, 180)

if __name__ == '__main__':
    import src.main as main

    freeze_support()

# TODO: for model, include has_disaster and season
# TODO: for point generation/data summary, use all disasters within the hexagon, not a point

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
    g. number of injured           EM-DAT
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



Climate parameters
Mess around with hexagon resolution
Get population data

"""