import pickle
import h3
import pandas as pd
import numpy as np
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet, QuantileLoss, RMSE, Baseline, GroupNormalizer, MultiLoss
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting.data.encoders import MultiNormalizer, TorchNormalizer
import torch
from src.configuration.config import Config
import os

import logging
import warnings
warnings.filterwarnings("ignore") 
logging.basicConfig(level=logging.INFO)

from src.utility import latlon_to_xyz

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

        self.continuous_targets = ['num_deaths', 'num_injured', 'damage_cost']
        self.binary_targets = ['has_landslide', 'has_flood', 'has_dry_mass_movement', 'has_extreme_temperature', 'has_storm', 'has_drought']
        # self.all_targets = self.continuous_targets + self.binary_targets
        
        # Defining loss metrics and normalizers for targets
        self.loss_metrics = dict()
        self.normalizers = dict()

        for target in self.continuous_targets:
            self.loss_metrics.update({
                target: QuantileLoss(
                    quantiles=[0.1, 0.5, 0.9]
                )
            })
            self.normalizers.update({
                target: GroupNormalizer(
                    groups=['group_id'],
                    transformation="log1p",
                    center=False,
                    scale_by_group=True
                )
            })

        # for target in self.binary_targets:
        #     self.loss_metrics.update({
        #         target: nn.BCEWithLogitsLoss()
        #     })
        #     self.normalizers.update({
        #         target: GroupNormalizer(
        #             groups=['group_id'],
        #             transformation=None,
        #             center=False,
        #             scale_by_group=False
        #         )
        #     })

        self.unknown_climate_reals = [
            "avg_temperature_2m",
            "min_temperature_2m",
            "max_temperature_2m",
            "avg_dewfrostpoint_2m",
            # "min_dewfrostpoint_2m",
            # "max_dewfrostpoint_2m",
            "avg_precipitation",
            # "min_precipitation",
            # "max_precipitation",
            "avg_windspeed_2m",
            # "min_windspeed_2m",
            # "max_windspeed_2m",
            "avg_windspeed_10m",
            # "min_windspeed_10m",
            # "max_windspeed_10m",
            "avg_windspeed_50m",
            # "min_windspeed_50m",
            # "max_windspeed_50m",
            "avg_humidity_2m",
            # "min_humidity_2m",
            # "max_humidity_2m",
            "avg_surface_pressure",
            # "min_surface_pressure",
            # "max_surface_pressure",
            "avg_transpiration",
            # "min_transpiration",
            # "max_transpiration",
            "avg_evaporation",
            # "min_evaporation",
            # "max_evaporation"
        ]
        
        df['timestamp'] = df['timestamp'].astype(int)
        df['has_disaster'] = df[self.binary_targets].any(axis=1).astype(float)
        # df[self.binary_targets] = df[self.binary_targets].astype(int).astype(float)
        df.drop(self.binary_targets, axis=1, inplace=True)

        # removes any duplicates
        df = self.remove_and_check_duplicates(df)

        # # data analysis
        # for continuous_target in self.continuous_targets:
        #     group_stats = df.groupby("group_id")[continuous_target].agg(["min", "max", "std", "count"])
        #     logging.info(group_stats)
        
        scaler = StandardScaler()       # apply standard scaling 
        df[['elevation']] = scaler.fit_transform(df[['elevation']])
        
        # make longitude and latitude learnable
        df['x'], df['y'], df['z'] = latlon_to_xyz(df['latitude'], df['longitude'])

        # splitting by date
        
        df.drop(['longitude', 'latitude'], axis=1, inplace=True)

        df.sort_values("timestamp").groupby("group_id")

        #training and validation set for model
        max_datapoints = df['timestamp'].max()
        train_end = int(Config.tft_train_split * max_datapoints)
        val_end = train_end + int(Config.tft_validation_split * max_datapoints)

        self.train_df = df[df['timestamp'] < train_end]
        self.val_df = df[(df['timestamp'] >= train_end) & (df['timestamp'] < val_end)]
        self.test_df = df[df['timestamp'] >= val_end]

        # Define max prediction & history length
        self.max_prediction_length = 52   # Forecast next 'x' weeks
        self.max_encoder_length = 520     # Use past 'x' weeks for prediction

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
                hidden_size=64, # 32
                learning_rate=0.03,
                batch_size=128,
                max_epochs=1000,
                patience=50
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

        # Create a MultiNormalizer for the targets
        target_normalizer = MultiNormalizer(
            # Order of dict.values() is preserved
            normalizers=list(self.normalizers.values())
        )

        tsds = TimeSeriesDataSet(
            data,
            time_idx='timestamp',
            target=self.continuous_targets, # self.all_targets,
            group_ids=['group_id'],  # Group by location for multi-region modeling
            min_encoder_length=max_encoder_length,
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            static_reals=['x','y','z', 'elevation'],  # Static features
            time_varying_known_reals=['timestamp'], # season?
            time_varying_unknown_reals=self.continuous_targets + self.unknown_climate_reals,
            target_normalizer = target_normalizer,
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

        loss = MultiLoss(
            # Order of dict.values() is preserved
            metrics=list(self.loss_metrics.values())
        )

        tft = TemporalFusionTransformer.from_dataset(
            train_dataset,
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
            trainer_kwargs=dict(limit_train_batches=30),
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
            'elevation': [encoder_data['elevation'].iloc[-1]] * prediction_length
        }

        for col in self.continuous_targets + self.unknown_climate_reals:
            decoder_data[col] = [0.] * prediction_length

        decoder_df = pd.DataFrame(decoder_data)

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
            mode="raw", # prediction
            return_x=True, 
            trainer_kwargs=dict(accelerator=Config.tft_accelerator)
        )
        logging.info("Raw predictions:")
        logging.info(raw_predictions)

        output = raw_predictions.output
        # logging.info("Raw output:")
        # logging.info(raw_output)
        x = raw_predictions.x
        # logging.info("x:")
        # logging.info(x)

        # predictions = output

        # predictions = best_tft.predict(
        #     data=location_tsds, 
        #     mode="prediction",
        #     trainer_kwargs=dict(accelerator=Config.tft_accelerator)
        # )

        predictions = {}
    
        # Extract and denormalize continuous targets
        for idx, target in enumerate(self.continuous_targets):
            normalized_pred = np.array(output.prediction[idx])
            
            # Get the normalizer for this target
            normalizer: GroupNormalizer = self.normalizers[target]
            
            # Extract scale and offset from x (these are added by add_target_scales=True)
            logging.info("Target scale")
            logging.info(x['target_scale'])

            scale = x["target_scale"][..., tsds.target_names.index(target)]
            offset = x["target_scale"][..., len(tsds.target_names) + tsds.target_names.index(target)]
            
            # Denormalize
            if normalizer.transformation == "log1p":
                # For log1p transformation: first apply scale and offset, then expm1
                denormalized_pred = np.expm1(normalized_pred * scale + offset)
            else:
                # For other transformations
                denormalized_pred = normalized_pred * scale + offset
                
            predictions[target] = denormalized_pred
        
        # # Extract binary targets (probabilities)
        # for target in self.binary_targets:
        #     # For binary targets, we get probabilities from sigmoid activation
        #     prob = torch.sigmoid(raw_predictions[target].prediction).detach().cpu().numpy()
        #     predictions[target] = prob
        #     # Also add binary predictions (0/1) using 0.5 threshold
        #     predictions[f"{target}_binary"] = (prob > 0.5).astype(int)

        # predictions_df = pd.DataFrame(predictions)
        
        return predictions
        
    def predict_test(self):
        counts = self.train_df[self.train_df["has_disaster"] == 1.0].groupby("group_id").size()
        group_id = counts.idxmax()
        logging.info(f"{group_id}'s max count: {counts.max()}")

        prev_df = pd.concat([self.train_df, self.val_df])
        prev_df = prev_df[prev_df['group_id'] == group_id]
        results = self.predict(location_data=prev_df)

        logging.info("results:")
        logging.info(results)

        analysis_df = self.test_df.copy()
        analysis_df = analysis_df[analysis_df['group_id'] == group_id]
        df_to_analyze = self.test_df.head(self.max_prediction_length)



        pass

        
        
    # TODO: not working yet
    def predict_and_digest(self, longitude: float, latitude: float):
        group_id = h3.str_to_int(
            h3.latlng_to_cell(
                lat=latitude, 
                lng=longitude, 
                res=Config.hexagon_resolution
            )
        )

        location_df = pd.read_parquet(
            path=Config.summary_dataset_filepath, 
            engine="pyarrow",
            filters=[('group_id', '==', group_id)]
        )

        # using 'recursive forecasting' for prediction past max_prediction_len

        prediction_df = pd.DataFrame(
            data=None,
            columns=['timestamp', *self.all_targets]
        )

        predictions = self.predict(location_data=location_df)

        return prediction_df

    def remove_and_check_duplicates(self, df):
        def get_duplicates(data):
            points = set([(long, lat) for long, lat in zip(data['longitude'], data['latitude'])])
            id_amounts = dict()
            for long, lat in points:
                group_id = h3.str_to_int(
                    h3.latlng_to_cell(
                        lat=lat, 
                        lng=long, 
                        res=Config.hexagon_resolution
                    )
                )
                new_list = id_amounts.get(group_id, []) + [(long, lat)]
                id_amounts.update({group_id: new_list})
            return id_amounts
        
        # removing duplicates
        id_amounts = get_duplicates(df)
        new_df = df.copy()
        duplicates_removed = 0
        for key, points in id_amounts.items():
            if len(points) > 1:
                for point in points[1:]:
                    new_df = new_df[~((new_df['longitude'] == point[0]) & (new_df['latitude'] == point[1]))]
                    duplicates_removed += 1
        print(f"{duplicates_removed} duplicates removed.")

        # double checking for duplicates
        id_amounts = get_duplicates(new_df)
        for key, points in id_amounts.items():
            if len(points) > 1:
                print(f"{key} has {points}")
        print("Done checking for duplicates.")

        return new_df




