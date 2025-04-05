# TODO: perhaps use this class for quality of life

from datetime import datetime

class Config:

    # TODO: create config vars for creating the disaster db

    ### disaster database

    furthest_back_time = datetime(1981, 1, 1)
    recreate_disaster_database = False
    show_disasters = False # TODO: implement
    # TODO: also maybe add vars and stuff to show the plots

    ### summary dataset

    # main driver for summary dataset:
    generate_summary_dataset = True

    recreate_summary_dataset = True # if True, all data from summary dataset will be cleared before processing
    start_date = datetime(1981, 1, 1) # earliest is 1981
    end_date = datetime(1990, 1, 1) # latest is 2018
    pointgen_instep = 5.0
    pointgen_circumstep = 7.0
    pointgen_absorb_radius = 0.0
    show_points = True # shows all points on world map
    num_workers = 2 # number of workers used to make data for summary dataset
