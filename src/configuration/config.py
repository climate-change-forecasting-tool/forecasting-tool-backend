
from datetime import datetime
from typing import Literal

class Config:

    ### filepaths

    disaster_gdb_filepath = "data/pend-gdis-1960-2018-disasterlocations-gdb/pend-gdis-1960-2018-disasterlocations.gdb"
    emdat_data_filepath = "data/emdat_disasterdata.xlsx"
    landmass_filepath = "data/ne_10m_land"
    combined_disaster_data_filepath = "db/disasters.parquet"
    summary_dataset_filepath = "db/summary_data.parquet"
    tft_checkpoint_path = "checkpoints/epoch=3-val_loss=0.06.ckpt"

    ### disaster database

    furthest_back_time = datetime(1981, 1, 1) # DO NOT MODIFY!!!

    recreate_disaster_database = False
    hexagon_resolution = 2
    show_hexagons = False
    show_disasters = False # is very slow

    ### summary dataset

    generate_summary_dataset = False # main driver for summary dataset

    recreate_summary_dataset = False # if True, all data from summary dataset will be cleared before processing
    start_date = datetime(1981, 1, 4) # earliest is 1981; ENSURE IT IS A SUNDAY
    end_date = datetime(2018, 12, 29) # latest is 2018; ENSURE IT IS A SATURDAY
    show_points = False # shows all points on world map
    num_workers = 4 # number of workers used to make data for summary dataset

    ### ML model

    activate_tft = True
    benchmark_tft = False
    tune_hyperparams_tft = False
    train_tft = True
    test_tft = False
    tft_accelerator: Literal['cpu', 'gpu', 'tpu', 'auto'] = 'cpu' 
    tft_validation_workers = 0 # max 8 if gpu is used; 0 if cpu
    tft_training_workers = 0 # max 4 if gpu is used; 0 if cpu
    tft_train_split = 0.6
    tft_validation_split = 0.2