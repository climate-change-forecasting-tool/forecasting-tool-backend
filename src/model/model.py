import pandas as pd
import numpy as np
from pytorch_forecasting import MAE, RMSE, Baseline, CrossEntropy, GroupNormalizer, MultiLoss, NaNLabelEncoder, QuantileLoss
from sklearn.preprocessing import StandardScaler
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import dask.dataframe as dd
from pytorch_forecasting.data.encoders import TorchNormalizer, MultiNormalizer
from src.configuration.config import Config

import logging

logging.basicConfig(level=logging.INFO)

# https://github.com/sktime/pytorch-forecasting/issues/359

class TFTransformer: # datalist, column_names, 
    def __init__(self, parquet_df, train_split: float = 0.6, validation_split: float = 0.2):
        """
        dataset_filename: a csv file
            expected to have:
            - timestamp
            - latitude
            - longitude
        """

        df = pd.read_parquet(path=parquet_df, engine="pyarrow")

        df['disastertype'] = df['disastertype'].astype('category')

        logging.info(df.columns)

        df['timestamp'] = df['timestamp'].astype(int)

        # apply standard scaling 
        scaler = StandardScaler()
        df[['elevation', 'num_deaths', 'num_injured', 'damage_cost']] = scaler.fit_transform(df[['elevation', 'num_deaths', 'num_injured', 'damage_cost']])

        # make longitude and latitude learnable

        def latlon_to_xyz(lat, lon):
            lat_rad = np.radians(lat)
            lon_rad = np.radians(lon)
            x = np.cos(lat_rad) * np.cos(lon_rad)
            y = np.cos(lat_rad) * np.sin(lon_rad)
            z = np.sin(lat_rad)
            return x, y, z

        df['x'], df['y'], df['z'] = latlon_to_xyz(df['latitude'], df['longitude'])


        # TODO: split by date or by ***location***

        # splitting by date
        max_datapoints = df['timestamp'].max()

        df.drop(['longitude', 'latitude'], axis=1, inplace=True)

        df.sort_values(by=['timestamp', 'x', 'y', 'z'], inplace=True)

        train_end = int(train_split*max_datapoints)
        val_end = train_end + int(validation_split*max_datapoints)

        self.train_df = df[df['timestamp'] < train_end]
        self.val_df = df[(df['timestamp'] >= train_end) & (df['timestamp'] < val_end)]
        self.test_df = df[df['timestamp'] >= val_end]

        # Define max prediction & history length
        self.max_prediction_length = 7   # Forecast next 'x' days
        self.max_encoder_length = 14     # Use past 'x' days for prediction


        # Can speed up in-memory operations of TimeSeriesDataSet with:
        # https://pypi.org/project/numba/

        train_dataset = self.prepare_multi_target_dataset(
            self.train_df,
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            predict_mode=True
        )

        val_dataset = self.prepare_multi_target_dataset(
            pd.concat([self.train_df, self.val_df]),
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            predict_mode=False
        )

        # test_dataset = self.prepare_multi_target_dataset(
        #     self.test_df,
        #     max_encoder_length=self.max_encoder_length,
        #     max_prediction_length=self.max_prediction_length,
        #     predict_mode=False
        # )

        ##### TRAINING
        if Config.train_tft:
            self.train(
                training_data=train_dataset,
                validation_data=val_dataset,
                hidden_size=32,
                learning_rate=0.03,
                batch_size=64,
                max_epochs=50
            )


        ###### hyperparameter tuning
        # self.tune_hyperparameters()

        ###### benchmark
        # logging.info("Benchmark:")
        # self.get_benchmark()

        ##### PERFORMANCE EVALUATION
        # self.eval_performance()

    def prepare_multi_target_dataset(
        self, 
        data, 
        max_encoder_length=24, 
        max_prediction_length=12,
        predict_mode=False
    ):
        # target_cols = ["num_deaths", "num_injured", "damage_cost", "disastertype"]

        # Prepare the data for categorical encoding
        categorical_encoder = NaNLabelEncoder(add_nan=True).fit(data["disastertype"])

        # Create a MultiNormalizer for the targets
        # Important: We need to fit it to our data before using it
        target_normalizer = MultiNormalizer(
            normalizers=[
                GroupNormalizer(groups=['x', 'y', 'z']),  # num_deaths
                GroupNormalizer(groups=['x', 'y', 'z']),  # num_injured
                GroupNormalizer(groups=['x', 'y', 'z']),  # damage_cost
                categorical_encoder,  # disastertype - no normalization for categorical variables
            ]
        )

        categorical_encoders = {
            "cat_target": categorical_encoder
        }

        tsds = TimeSeriesDataSet(
            data,
            time_idx='timestamp',
            target=['num_deaths', 'num_injured', 'damage_cost', 'disastertype'], # maybe make disastertype one-hot real
            group_ids=['x', 'y', 'z'],  # Group by location for multi-region modeling
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            static_reals=['x', 'y', 'z', 'elevation'],  # Static features
            time_varying_unknown_categoricals=['disastertype'],
            time_varying_known_reals=['timestamp'], # season?
            time_varying_unknown_reals=[
                "num_deaths",
                "num_injured",
                "damage_cost",
                "avg_temperature_2m",
                "min_temperature_2m",
                "max_temperature_2m",
                "dewfrostpoint_2m",
                "precipitation",
                "avg_windspeed_2m",
                "min_windspeed_2m",
                "max_windspeed_2m",
                "avg_windspeed_10m",
                "min_windspeed_10m",
                "max_windspeed_10m",
                "avg_windspeed_50m",
                "min_windspeed_50m",
                "max_windspeed_50m",
                "humidity_2m",
                "surface_pressure",
                "transpiration",
                "evaporation"
            ],
            categorical_encoders=categorical_encoders,
            target_normalizer = target_normalizer,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            randomize_length=False,
            predict_mode=predict_mode
        )

        return tsds

    def load_best_model(self):
        # load the best model according to the validation loss
        best_model_path = self.trainer.checkpoint_callback.best_model_path
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        return best_tft
    
    def tune_hyperparameters(self):
        ###### hyperparameter tuning
        import pickle

        from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

        # create study
        study = optimize_hyperparameters(
            self.train_loader,
            self.val_loader,
            model_path="optuna_test",
            n_trials=200,
            max_epochs=50,
            gradient_clip_val_range=(0.01, 1.0),
            hidden_size_range=(8, 128),
            hidden_continuous_size_range=(8, 128),
            attention_head_size_range=(1, 4),
            learning_rate_range=(0.001, 0.1),
            dropout_range=(0.1, 0.3),
            trainer_kwargs=dict(limit_train_batches=30),
            reduce_on_plateau_patience=4,
            use_learning_rate_finder=False,  # Optuna for analyzing ideal learning rate or use in-built learning rate finder
        )

        # save study results - also we can resume tuning at a later point in time
        with open("test_study.pkl", "wb") as fout:
            pickle.dump(study, fout)

        # show best hyperparameters
        logging.info(study.best_trial.params)

    def get_benchmark(self):
        baseline_predictions = Baseline().predict(self.val_loader, return_y=True)
        logging.info(baseline_predictions)
        logging.info(MAE()(baseline_predictions.output, baseline_predictions.y))

    def eval_performance(self):
        best_tft = self.load_best_model()

        # calculate mean absolute error on validation set
        # predictions = best_tft.predict(
        #     self.val_loader, return_y=True, trainer_kwargs=dict(accelerator="cpu")
        # )
        predictions = best_tft.predict(
            self.test_loader, return_y=True, trainer_kwargs=dict(accelerator="cpu")
        )
        logging.info(MAE()(predictions.output, predictions.y)) # Mean average absolute error

    def train(
        self,
        training_data: TimeSeriesDataSet, 
        validation_data: TimeSeriesDataSet = None, 
        hidden_size=32, 
        learning_rate=0.03, 
        batch_size=32, 
        max_epochs=50
    ):

        self.train_loader = training_data.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        self.val_loader = validation_data.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

        # Setup trainer
        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
        )
        lr_logger = LearningRateMonitor()  # log learning rate
        logger = TensorBoardLogger("lightning_logs")

        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            enable_model_summary=True,
            gradient_clip_val=0.1,
            limit_train_batches=50,  # comment in for training, running valiation every 30 batches
            fast_dev_run=True,
            callbacks=[lr_logger, early_stop_callback],
            logger=logger,
        )

        loss = MultiLoss([
            QuantileLoss(),  # num_deaths
            QuantileLoss(),  # num_injured
            QuantileLoss(),  # damage_cost
            CrossEntropy()   # disastertype
        ])

        self.tft = TemporalFusionTransformer.from_dataset(
            training_data,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=8,
            loss=loss,
            log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
            optimizer="ranger",
            reduce_on_plateau_patience=4,
        )
        logging.info(f"Number of parameters in network: {self.tft.size() / 1e3:.1f}k")

        self.trainer.fit(
            self.tft,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader,
        )

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