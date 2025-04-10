# TODO: perhaps use this class for quality of life

from datetime import datetime

class Config:

    ### filepaths

    disaster_gdb_filepath = "data/pend-gdis-1960-2018-disasterlocations-gdb/pend-gdis-1960-2018-disasterlocations.gdb"
    emdat_data_filepath = "data/emdat_disasterdata.xlsx"
    landmass_filepath = "data/ne_10m_land"
    combined_disaster_data_filepath = "db/disasters.parquet"
    summary_dataset_filepath = "db/summary_data.parquet"

    ### disaster database

    furthest_back_time = datetime(1981, 1, 1)

    recreate_disaster_database = False
    hexagon_resolution = 2
    show_hexagons = True
    show_disasters = False # is very slow

    ### summary dataset

    generate_summary_dataset = False # main driver for summary dataset

    recreate_summary_dataset = False # if True, all data from summary dataset will be cleared before processing
    start_date = datetime(1981, 1, 1) # earliest is 1981
    end_date = datetime(1990, 1, 1) # latest is 2018
    show_points = True # shows all points on world map
    num_workers = 4 # number of workers used to make data for summary dataset

    ### ML model

    activate_tft = True
    train_tft = True
    test_tft = False
    # TODO: perhaps put all model parameters here