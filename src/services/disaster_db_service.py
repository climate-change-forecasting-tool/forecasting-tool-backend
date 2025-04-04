from src.controllers.disaster_db_controller import DisasterDBController
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import itertools

import logging
logging.basicConfig(level=logging.INFO)

DB_FILE_PATH = "data/disasters.db"

class DisasterDBService:
    def __init__(self):
        self.controller = DisasterDBController()

    




