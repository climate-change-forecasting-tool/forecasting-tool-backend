
from datetime import datetime
from typing import Literal

class Config:

    climate_params_shortnames = [
        't2m',  # temperature 2m; K
        'u10',  # wind speed 10m u-comp; m/s
        'v10',  # wind speed 10m v-comp; m/s
        'lsm', # land sea mask; 0-1
        'sp', # Surface pressure; Pa
        'aod550', # Total aerosol optical depth at 550 nm; ~
        'tc_ch4', # Total column methane; kg/m^2
        'tcno2', # Total column nitrogen dioxide; kg/m^2
        'tco3' # Total column ozone; kg/m^2
        'tcso2', # Total column sulphur dioxide; kg/m^2
        'tcwv' # Total column water vapour; kg/m^2
    ]

    climate_data_param_names = dict({
        '2m_temperature': 't2m',  # temperature 2m; K
        '10m_u_component_of_wind': 'u10',  # wind speed 10m u-comp; m/s
        '10m_v_component_of_wind': 'v10',  # wind speed 10m v-comp; m/s
        'land_sea_mask': 'lsm', # land sea mask; 0-1
        'surface_pressure': 'sp', # Surface pressure; Pa
        'total_aerosol_optical_depth_550nm': 'aod550', # Total aerosol optical depth at 550 nm; ~
        'total_column_methane': 'tc_ch4', # Total column methane; kg/m^2
        'total_column_nitrogen_dioxide': 'tcno2', # Total column nitrogen dioxide; kg/m^2
        'total_column_ozone': 'tco3', # Total column ozone; kg/m^2
        'total_column_sulphur_dioxide': 'tcso2', # Total column sulphur dioxide; kg/m^2
        'total_column_water_vapour': 'tcwv', # Total column water vapour; kg/m^2
    })

    ### filepaths

    landmass_filepath = "data/ne_10m_land"
    climate_data_output_dir = "data/climate_data/"
    partitioned_climate_data_dir = "db/partitioned_data/"
    summary_dataset_filepath = "db/summary_data.parquet"
    tft_checkpoint_path = "checkpoints/epoch=3-val_loss=0.06.ckpt"

    ### climate data

    download_climate_data = False

    ### summary dataset

    hexagon_resolution = 2
    show_init_hexagons = False
    show_post_hexagons = False
    generate_summary_dataset = True # main driver for summary dataset
    recreate_summary_dataset = True # if True, all data from summary dataset will be cleared before processing
    start_date = datetime(2003, 1, 1) # earliest is datetime(2003, 1, 1)
    end_date = datetime(2003, 1, 31) # latest is datetime(2024, 12, 31)

    ### ML model

    activate_tft = False
    benchmark_tft = False
    tune_hyperparams_tft = False
    train_tft = False
    test_tft = False
    tft_accelerator: Literal['cpu', 'gpu', 'tpu', 'auto'] = 'cpu' 
    tft_validation_workers = 0 # max 8 if gpu is used; 0 if cpu
    tft_training_workers = 0 # max 4 if gpu is used; 0 if cpu
    tft_train_split = 0.6
    tft_validation_split = 0.2