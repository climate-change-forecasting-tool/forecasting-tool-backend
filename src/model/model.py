import warnings
warnings.filterwarnings("ignore")  # avoid printing out absolute paths

import pandas as pd
import numpy as np
from pytorch_forecasting import MAE, Baseline, CrossEntropy, GroupNormalizer, MultiLoss, NaNLabelEncoder, QuantileLoss
from sklearn.preprocessing import StandardScaler
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import dask.dataframe as dd
from pytorch_forecasting.data.encoders import MultiNormalizer

from src.configuration.config import Config




import logging

logging.basicConfig(level=logging.INFO)

# https://github.com/sktime/pytorch-forecasting/issues/359

class TFTransformer: # datalist, column_names, 
    def __init__(self):
        """
        dataset_filename: a csv file
            expected to have:
            - timestamp
            - latitude
            - longitude
        """

        # make longitude and latitude learnable

    def latlon_to_xyz(lat, lon):
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)
        return x, y, z

    def load_best_model(self):
        # load the best model according to the validation loss
        best_model_path = self.trainer.checkpoint_callback.best_model_path
        logging.info(f"Loading {best_model_path}")
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        return best_tft

    def predict(self, location: tuple, timestamp):
        """
        location - a tuple containing (longitude, latitude)
        """

        best_tft = self.load_best_model()

        # select last 24 months from data (max_encoder_length is 24)
        encoder_data = self.df[lambda x: x.timestamp > x.timestamp.max() - self.max_encoder_length]

        # select last known data point and create decoder data from it
        last_data = self.df[lambda x: x.timestamp == x.timestamp.max()]
        decoder_data = pd.concat(
            [
                last_data.assign(date=lambda x: x.date + pd.offsets.MonthBegin(i))
                for i in range(1, self.max_prediction_length + 1)
            ],
            ignore_index=True,
        )

        # add time index consistent with "data"
        decoder_data["timestamp"] = (
            decoder_data["date"].dt.year * 12 + decoder_data["date"].dt.month
        )
        decoder_data["timestamp"] += (
            encoder_data["timestamp"].max() + 1 - decoder_data["timestamp"].min()
        )

        # combine encoder and decoder data
        new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)

        new_raw_predictions = best_tft.predict(
            new_prediction_data,
            mode="raw",
            return_x=True,
            trainer_kwargs=dict(accelerator="cpu"),
        )

        # for idx in range(10):  # plot 10 examples
        #     best_tft.plot_prediction(
        #         new_raw_predictions.x,
        #         new_raw_predictions.output,
        #         idx=idx,
        #         show_future_observed=False,
        #     )

        # ##### variable importance
        # interpretation = best_tft.interpret_output(raw_prediction.output, reduction="sum")
        # best_tft.plot_interpretation(interpretation)

        return new_raw_predictions