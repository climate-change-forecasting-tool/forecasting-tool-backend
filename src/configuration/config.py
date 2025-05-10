
from datetime import datetime
from typing import Literal

class Config:

    ### filepaths

    cams_files_and_vars = {
        "data/714aa50481381d97419b0527369f12ac.grib": [
            't2m',  # temperature 2m; K
            'u10',  # wind speed 10m u-comp; m/s
            'v10',  # wind speed 10m v-comp; m/s
        ],
        # "": [
        #     'lsm', # land sea mask; 0-1
        #     'sp', # Surface pressure; Pa
        #     'aod550', # Total aerosol optical depth at 550 nm; ~
        # ],
        # "": [
        #     'tc_ch4', # Total column methane; kg/m^2
        #     'tcno2', # Total column nitrogen dioxide; kg/m^2
        #     'tco3' # Total column ozone; kg/m^2
        # ],
        # "": [
        #     'tcso2', # Total column sulphur dioxide; kg/m^2
        #     'tcwv' # Total column water vapour; kg/m^2
        # ],
    }

    landmass_filepath = "data/ne_10m_land"
    summary_dataset_filepath = "db/summary_data.parquet"
    tft_checkpoint_path = "checkpoints/epoch=3-val_loss=0.06.ckpt"

    ### summary dataset

    hexagon_resolution = 2
    show_init_hexagons = True
    show_post_hexagons = True
    generate_summary_dataset = False # main driver for summary dataset
    recreate_summary_dataset = False # if True, all data from summary dataset will be cleared before processing
    start_date = datetime(2003, 1, 1) # earliest is datetime(2003, 1, 1)
    end_date = datetime(2003, 1, 31) # latest is datetime(2024, 12, 31)

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