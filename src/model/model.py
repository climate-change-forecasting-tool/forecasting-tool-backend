from typing import List
import pandas as pd
import numpy as np
from pytorch_forecasting import NaNLabelEncoder, TemporalFusionTransformer, TimeSeriesDataSet, QuantileLoss, RMSE, Baseline, GroupNormalizer
from sklearn.preprocessing import StandardScaler
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from src.configuration.config import Config
import os
from datetime import timedelta

import logging
logging.basicConfig(level=logging.INFO)
import warnings
warnings.filterwarnings("ignore") 
import optuna
optuna.logging.set_verbosity(verbosity=optuna.logging.INFO)

from src.utility import latlon_to_xyz, get_astronomical_season

class TFTransformer: 
    def __init__(self):
        df = pd.read_parquet(path=Config.summary_dataset_filepath, engine="pyarrow")

        self.checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints/",
            filename="{epoch}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            verbose=True
        )

        self.target = 't2m_mean'
        
        # Defining loss metrics and normalizers for targets
        self.loss_metrics = QuantileLoss(
            quantiles=[0.1, 0.5, 0.9]
        )
        self.normalizers = GroupNormalizer(
            groups=['group_id'],
            method="standard"
        )

        self.unknown_climate_reals: List[str] = [
            't2m_min', 
            't2m_mean', 
            't2m_max', 
            'u10_min', 
            'u10_mean', 
            'u10_max', 
            'v10_min', 
            'v10_mean', 
            'v10_max', 
            'lsm_mean', 
            'sp_mean', 
            'aod550_max', 
            'tc_ch4_max', 
            'tcno2_max', 
            'gtco3_max', 
            'tcso2_max', 
            'tcwv_max'
        ]
        
        self.unknown_climate_reals = list(set(self.unknown_climate_reals).difference([self.target]))
        
        df['season'] = df['season'].astype('category')

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = (df['timestamp'] - df['timestamp'].min()).dt.days.astype(int)

        # logging.info(df)
        
        # make longitude and latitude learnable
        df['x'], df['y'], df['z'] = latlon_to_xyz(df['latitude'], df['longitude'])

        # splitting by date
        df.drop(['longitude', 'latitude'], axis=1, inplace=True)

        df.sort_values(by="timestamp", inplace=True)

        #training and validation set for model
        max_datapoints = df['timestamp'].max()
        train_end = int(Config.tft_train_split * max_datapoints)
        val_end = train_end + int(Config.tft_validation_split * max_datapoints)

        self.train_df = df[df['timestamp'] < train_end]
        self.val_df = df[(df['timestamp'] >= train_end) & (df['timestamp'] < val_end)]
        self.test_df = df[df['timestamp'] >= val_end]

        scaler = StandardScaler()       # apply standard scaling 
        scaler.fit(self.train_df[self.unknown_climate_reals])

        self.data_means = scaler.mean_
        self.data_stds = scaler.var_

        self.train_df[self.unknown_climate_reals] = scaler.transform(self.train_df[self.unknown_climate_reals])
        self.val_df[self.unknown_climate_reals] = scaler.transform(self.val_df[self.unknown_climate_reals])
        self.test_df[self.unknown_climate_reals] = scaler.transform(self.test_df[self.unknown_climate_reals])

        # logging.info("The datasets:")
        # logging.info(self.train_df)
        # logging.info(self.val_df)
        # logging.info(self.test_df)

        # Define max prediction & history length
        self.max_prediction_length = 14   # Forecast next 'x' days; 14
        self.max_encoder_length = 365     # Use past 'x' days for prediction; 365

        if Config.benchmark_tft:
            logging.info("Benchmark:")
            self.get_benchmark(
                training_dataframe=self.train_df,
                validation_dataframe=self.val_df,
                batch_size=128
            )

        ###### HYPERPARAMETER TUNING
        if Config.tune_hyperparams_tft:
            logging.info("Hyperparameter tuning:")
            self.tune_hyperparameters(
                training_dataframe=self.train_df,
                validation_dataframe=self.val_df,
                batch_size=128,
            )

        ##### TRAINING
        if Config.train_tft:
            logging.info("Training:")
            self.train(
                training_dataframe=self.train_df,
                validation_dataframe=self.val_df,
                hidden_size=32, # 32
                learning_rate=0.03,
                batch_size=128,
                max_epochs=1000,
                patience=25
            )

        if Config.test_tft:
            logging.info("Predicting test data:")
            self.predict_test()

        ##### PERFORMANCE EVALUATION
        # self.eval_performance()

    def prepare_multi_target_dataset(
        self, 
        data, 
        max_encoder_length=24, 
        max_prediction_length=12,
        predict_mode=False
    ):

        tsds = TimeSeriesDataSet(
            data,
            time_idx='timestamp',
            target=self.target,
            group_ids=['group_id'],  # Group by location for multi-region modeling
            min_encoder_length=max_encoder_length,
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            time_varying_known_categoricals=['season'],
            static_reals=['x','y','z'],  # Static features
            time_varying_known_reals=['timestamp'], # season?
            time_varying_unknown_reals=[self.target] + self.unknown_climate_reals,
            target_normalizer = self.normalizers,
            categorical_encoders={
                "season": NaNLabelEncoder().fit(data["season"])
            },
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            randomize_length=False,
            predict_mode=predict_mode
        )
        return tsds

    def train(
        self,
        training_dataframe: pd.DataFrame, 
        validation_dataframe: pd.DataFrame = None, 
        hidden_size=32, 
        learning_rate=0.03, 
        batch_size=32, 
        max_epochs=50,
        patience=10,
    ):
        train_dataset = self.prepare_multi_target_dataset(
            training_dataframe,
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            predict_mode=False
        )

        val_dataset = self.prepare_multi_target_dataset(
            pd.concat([training_dataframe, validation_dataframe]),
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            predict_mode=True
        )

        train_loader = train_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=Config.tft_training_workers)
        val_loader = val_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=Config.tft_validation_workers)

        # Setup trainer
        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=1e-4, patience=patience, verbose=False, mode="min"
        )
        # lr_logger = LearningRateMonitor()  # log learning rate
        
        logger = TensorBoardLogger("lightning_logs")

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=Config.tft_accelerator,
            enable_model_summary=True,
            gradient_clip_val=0.1,
            limit_train_batches=50,  # comment in for training, running valiation every 30 batches
            # fast_dev_run=True,
            callbacks=[early_stop_callback, self.checkpoint_callback],  # Added checkpoint_callback here
            logger=logger,
        )

        tft = TemporalFusionTransformer.from_dataset(
            train_dataset,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=8,
            loss=self.loss_metrics,
            log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
            optimizer="ranger",
            reduce_on_plateau_patience=4,
        )
        logging.info(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

        trainer.fit(
            tft,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        # with open("checkpoints/output_transformer.pkl", "wb") as f:
        #     pickle.dump(self.tft.output_transformer, f)

    def load_best_model(self, checkpoint_path=None):
        """
        Load the best model from a checkpoint
        Parameters:  checkpoint_path : str, optional
            Path to a specific checkpoint file. If None, attempts to find the best checkpoint
            from the trainer if training has occurred.
        Returns: The loaded model
        """
        if checkpoint_path is None:
            logging.info("Trying to get best model from checkpoints folder...")
            # Get all .ckpt files
            ckpt_files = [f for f in os.listdir("checkpoints") if f.endswith(".ckpt")]

            # Sort by val_loss parsed from filename
            ckpt_files.sort(key=lambda f: float(f.split("val_loss=")[-1].replace(".ckpt", "")))

            # Select best
            best_checkpoint = ckpt_files[0]
            checkpoint_path = os.path.join("checkpoints", best_checkpoint)

        if checkpoint_path is None:
            logging.info("Trying to get best model from predefined path...")
            checkpoint_path = Config.tft_checkpoint_path
        
        # Verify the checkpoint file exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        logging.info(f"Using best model from: {checkpoint_path}")

        # Load the model
        best_tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)

        return best_tft
    
    def model_to_cpu(self, checkpoint_path: str = None):
        if checkpoint_path is None:
            logging.info("Trying to get best model from checkpoints folder...")
            # Get all .ckpt files
            ckpt_files = [f for f in os.listdir("checkpoints") if f.endswith(".ckpt")]

            # Sort by val_loss parsed from filename
            ckpt_files.sort(key=lambda f: float(f.split("val_loss=")[-1].replace(".ckpt", "")))

            # Select best
            best_checkpoint = ckpt_files[0]
            checkpoint_path = os.path.join("checkpoints", best_checkpoint)

        if checkpoint_path is None:
            logging.info("Trying to get best model from predefined path...")
            checkpoint_path = Config.tft_checkpoint_path
        
        # Verify the checkpoint file exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        logging.info(f"Using best model from: {checkpoint_path}")

        checkpoint_model = TemporalFusionTransformer.load_from_checkpoint(
            checkpoint_path=checkpoint_path, 
            map_location=torch.device("cpu")
        )

        new_checkpoint_path = "checkpoints/model.ckpt"
        torch.save(checkpoint_model, new_checkpoint_path)
    
    def tune_hyperparameters(
        self,
        training_dataframe: pd.DataFrame, 
        validation_dataframe: pd.DataFrame = None, 
        batch_size=32,
        n_trials = 200, 
        max_epochs=50,
    ):
        ###### hyperparameter tuning
        import pickle
        from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

        training_data = self.prepare_multi_target_dataset(
            training_dataframe,
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            predict_mode=False
        )

        validation_data = self.prepare_multi_target_dataset(
            pd.concat([training_dataframe, validation_dataframe]),
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            predict_mode=True
        )

        train_loader = training_data.to_dataloader(train=True, batch_size=batch_size, num_workers=Config.tft_training_workers)
        val_loader = validation_data.to_dataloader(train=False, batch_size=batch_size, num_workers=Config.tft_validation_workers)

        # create study
        study = optimize_hyperparameters(
            train_loader,
            val_loader,
            model_path="optuna_test",
            n_trials=n_trials,
            max_epochs=max_epochs,
            gradient_clip_val_range=(0.01, 1.0),
            hidden_size_range=(8, 128),
            hidden_continuous_size_range=(8, 128),
            attention_head_size_range=(1, 4),
            learning_rate_range=(0.001, 0.1),
            dropout_range=(0.1, 0.3),
            trainer_kwargs=dict(limit_train_batches=30, enable_progress_bar=True),
            reduce_on_plateau_patience=4,
            use_learning_rate_finder=False,  # Optuna for analyzing ideal learning rate or use in-built learning rate finder
            verbose=True
        )

        # save study results - also we can resume tuning at a later point in time
        with open("test_study.pkl", "wb") as fout:
            pickle.dump(study, fout)

        # show best hyperparameters
        logging.info(study.best_trial.params)

    def get_benchmark(
        self, 
        training_dataframe: pd.DataFrame,
        validation_dataframe: pd.DataFrame = None,
        batch_size=32,
    ):
        val_dataset = self.prepare_multi_target_dataset(
            pd.concat([training_dataframe, validation_dataframe]),
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            predict_mode=True
        )
        val_loader = val_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=Config.tft_validation_workers)
        
        baseline_predictions = Baseline().predict(val_loader, return_y=True)
        logging.info(baseline_predictions)
        logging.info(RMSE()(baseline_predictions.output, baseline_predictions.y))

    def eval_performance(
        self, 
        training_dataframe: pd.DataFrame,
        validation_dataframe: pd.DataFrame = None,
        batch_size = 32,
    ):
        best_tft = self.load_best_model()

        val_dataset = self.prepare_multi_target_dataset(
            pd.concat([training_dataframe, validation_dataframe]),
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            predict_mode=True
        )
        val_loader = val_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=Config.tft_validation_workers)
        
        predictions = best_tft.predict(
            val_loader, return_y=True, trainer_kwargs=dict(accelerator=Config.tft_accelerator)
        )
        logging.info(RMSE()(predictions.output, predictions.y))

    # TODO: test and fix
    def predict(self, location_data: pd.DataFrame, prediction_length: int = None):
        """
        Generate predictions for future time steps based on location data.
        Parameters:
        location_data : pd.DataFrame
            DataFrame containing the most recent data for a specific location to use as context.
            Must include the same features used during training.
        prediction_length : int, optional
            Number of time steps to predict into the future. If None, uses self.max_prediction_length.

        Returns:dict : Dictionary containing:
            - 'predictions': Raw prediction outputs with quantiles for numerical targets
            - 'disaster_probabilities': Probabilities for each disaster type
            - 'predicted_values': Point estimates for numerical targets (deaths, injuries, damage)
        """
        if prediction_length is None:
            prediction_length = self.max_prediction_length
        
        # Ensure we have enough history for encoder
        if len(location_data) < self.max_encoder_length:
            raise ValueError(f"Input data must contain at least {self.max_encoder_length} time steps for encoding")
        
        # Get the most recent data for encoding context
        encoder_data = location_data.tail(self.max_encoder_length).copy()

        # Load the best model
        best_tft = self.load_best_model()

        # Create future timestamps for prediction
        last_timestamp = encoder_data['timestamp'].max()
        future_timestamps = [last_timestamp + i + 1 for i in range(prediction_length)]

        decoder_data = {
            'timestamp': future_timestamps,
            'group_id': [encoder_data['group_id'].iloc[0]] * prediction_length,
            'x': [encoder_data['x'].iloc[0]] * prediction_length,
            'y': [encoder_data['y'].iloc[0]] * prediction_length,
            'z': [encoder_data['z'].iloc[0]] * prediction_length,
        }

        for col in [self.target] + self.unknown_climate_reals:
            decoder_data[col] = [0.] * prediction_length

        future_datetimes = [Config.start_date + timedelta(days=float(d)) for d in future_timestamps]
        decoder_data['season'] = [
            get_astronomical_season(
                date=future_dt, 
                latitude=np.rad2deg(np.arcsin(z_val)) # gives latitude
            ) 
            for future_dt, z_val in zip(future_datetimes, decoder_data['z'])
        ]

        decoder_df = pd.DataFrame(decoder_data)
        decoder_df['season'] = decoder_df['season'].astype('category')

        combined_data = pd.concat([encoder_data, decoder_df])

        tsds = self.prepare_multi_target_dataset(
            data=combined_data,
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            predict_mode=True
        )

        dataloader = tsds.to_dataloader(train=False, batch_size=128, num_workers=Config.tft_validation_workers)
        
        raw_predictions = best_tft.predict(
            dataloader, # or tsds
            mode="raw",
            return_x=True, 
            trainer_kwargs=dict(accelerator=Config.tft_accelerator)
        )

        raw_output = raw_predictions.output.prediction
        # logging.info("Denormalized output:")
        # logging.info(raw_output)

        # x = raw_predictions.x

        predictions = {}
    
        # Extract and denormalize continuous targets
        for idx, target in enumerate([self.target]):
            denormalized_pred = np.array(raw_output[idx].cpu())
            
            logging.info("Denormalized")
            logging.info(denormalized_pred)

            # Get the normalizer for this target
            # normalizer: GroupNormalizer = self.normalizers[target]
            
            # Extract scale and offset from x (these are added by add_target_scales=True)
            # target_scale = np.array(x["target_scale"][idx].cpu()).flatten()

            # scale = target_scale[1]
            # offset = target_scale[0]

            # logging.info(f"Scale: {scale} | offset: {offset}")

            # logging.info("Normalized:")
            # logging.info((denormalized_pred - offset) / scale)

            denormalized_pred = np.sort(denormalized_pred, axis=1)

            median_idx = denormalized_pred.shape[1] // 2
            
            predictions[target + '_lower'] = denormalized_pred[:, 0]
            predictions[target + '_median'] = denormalized_pred[:, median_idx]
            predictions[target + '_upper'] = denormalized_pred[:, -1]

        predictions_df = pd.DataFrame(predictions)
        
        return predictions_df
    
    def evaluate(self):
        pass
        
    def predict_test(self):
        group_id = self.train_df.iloc[0]['group_id']

        prev_df = pd.concat([self.train_df, self.val_df])
        prev_df = prev_df[prev_df['group_id'] == group_id]
        results = self.predict(location_data=prev_df)

        logging.info("results:")
        logging.info(results)

        analysis_df = self.test_df.copy()
        analysis_df = analysis_df[analysis_df['group_id'] == group_id]
        df_to_analyze = self.test_df.head(self.max_prediction_length)
        


