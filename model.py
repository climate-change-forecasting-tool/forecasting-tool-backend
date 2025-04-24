import pandas as pd
import numpy as np
from pytorch_forecasting import MAE, RMSE, Baseline, QuantileLoss
from sklearn.preprocessing import StandardScaler
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

import logging

logging.basicConfig(level=logging.INFO)

class TFTransformer:
    def __init__(self, datalist, column_names, train_split: float = 0.6, validation_split: float = 0.2):
        """
        dataset_filename: a csv file
            expected to have:
            - timestamp
            - latitude
            - longitude
        """

        # Load your climate dataset (ensure it has a timestamp column)
        self.df = pd.DataFrame(data=datalist, columns=column_names)

        # logging.info(df)

        # Adjust timestamp to start from 0 and have an increment of 1 between contiguous times
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], format='%Y%m%d%H')
        self.df['timestamp'] = ((self.df['timestamp'] - self.df['timestamp'].min()).dt.total_seconds() // 3600).astype(int)

        # Sort data for time-series processing
        self.df = self.df.sort_values(['longitude', 'latitude', 'elevation', 'timestamp'])

        logging.info(self.df)

        # Normalize numerical features
        scaler = StandardScaler()
        self.df[['T2M', 'RH2M', 'WS2M', 'PRECTOTCORR']] = scaler.fit_transform(self.df[['T2M', 'RH2M', 'WS2M', 'PRECTOTCORR']])

        logging.info(self.df)

        # Define max prediction & history length
        self.max_prediction_length = 24   # Forecast next 'x' hours
        self.max_encoder_length = 48     # Use past 'x' hours for prediction

        train_end = int(train_split*len(self.df))
        val_end = train_end + int(validation_split*len(self.df))
        
        train_df = self.df[:train_end]
        val_df = self.df[train_end:val_end]
        test_df = self.df[val_end:]

        train_dataset = TimeSeriesDataSet(
            train_df,
            time_idx='timestamp',
            target='T2M',
            group_ids=['longitude', 'latitude'],  # Group by location for multi-region modeling
            min_encoder_length=self.max_encoder_length,
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            static_reals=['longitude', 'latitude', 'elevation'],  # Static features
            time_varying_known_reals=['timestamp', 'RH2M', 'WS2M', 'PRECTOTCORR'],  # Known variables
            time_varying_unknown_reals=['T2M'],  # Variables to predict
            # target_normalizer=GroupNormalizer(groups=['longitude', 'latitude', 'elevation']),  # Normalize per location
            target_normalizer=None
        )

        val_dataset = TimeSeriesDataSet.from_dataset(
            train_dataset, pd.concat([train_df, val_df]), predict=True, stop_randomization=True
        )

        batch_size = 64  # Adjust based on system memory

        self.train_loader = train_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        self.val_loader = val_dataset.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

        # Setup trainer
        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
        )
        lr_logger = LearningRateMonitor()  # log learning rate
        logger = TensorBoardLogger("lightning_logs")

        self.trainer = pl.Trainer(
            max_epochs=50,
            accelerator="cpu",
            enable_model_summary=True,
            gradient_clip_val=0.1,
            limit_train_batches=50,  # comment in for training, running valiation every 30 batches
            # fast_dev_run=True,
            callbacks=[lr_logger, early_stop_callback],
            logger=logger,
        )

        self.tft = TemporalFusionTransformer.from_dataset(
            train_dataset,
            learning_rate=0.01,
            hidden_size=64,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=8,
            loss=RMSE(), # QuantileLoss()
            log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
            optimizer="ranger",
            reduce_on_plateau_patience=4,
        )
        logging.info(f"Number of parameters in network: {self.tft.size() / 1e3:.1f}k")


        ##### TRAINING
        self.train()


        ###### hyperparameter tuning
        # self.tune_hyperparameters()

        ###### benchmark
        logging.info("Benchmark:")
        self.get_benchmark()

        ##### PERFORMANCE EVALUATION
        self.eval_performance()

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
        predictions = best_tft.predict(
            self.val_loader, return_y=True, trainer_kwargs=dict(accelerator="cpu")
        )
        logging.info(MAE()(predictions.output, predictions.y)) # Mean average absolute error

    def train(self):
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

