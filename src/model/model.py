import pickle
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
import torch
from src.configuration.config import Config

import logging
import warnings
warnings.filterwarnings("ignore") 
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
        df['timestamp'] = df['timestamp'].astype(int)
        
        scaler = StandardScaler()       # apply standard scaling 
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

        # splitting by date
        max_datapoints = df['timestamp'].max()
        #data organization
        row1 = df.iloc[0]
        long1 = row1['longitude']
        lat1 = row1['latitude']
        #df = df[(df['longitude'] == long1) & (df['latitude'] == lat1)] oneline
        df['group_id'] = df.groupby(['longitude', 'latitude']).ngroup()
        df.sort_values(by = ['timestamp', 'group_id'])
        df["disastertype"] = df["disastertype"].fillna('none')
        df.drop(['longitude', 'latitude'], axis=1, inplace=True)
        #training and validation set for model
        train_end = int(train_split*max_datapoints)
        val_end = train_end + int(validation_split*max_datapoints)

        self.train_df = df[df['timestamp'] < train_end]
        self.val_df = df[(df['timestamp'] >= train_end) & (df['timestamp'] < val_end)]
        self.test_df = df[df['timestamp'] >= val_end]

        # Define max prediction & history length
        self.max_prediction_length = 14   # Forecast next 'x' days
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
                GroupNormalizer(groups=['group_id']),  # num_deaths
                GroupNormalizer(groups=['group_id']),  # num_injured
                GroupNormalizer(groups=['group_id']),  # damage_cost
                categorical_encoder,  # disastertype - no normalization for categorical variables
            ]
        )

        categorical_encoders = {"disastertype": categorical_encoder}

        tsds = TimeSeriesDataSet(
            data,
            time_idx='timestamp',
            target=['num_deaths', 'num_injured', 'damage_cost', 'disastertype'], # maybe make disastertype one-hot real
            group_ids=['group_id'],  # Group by location for multi-region modeling
            min_encoder_length=max_encoder_length,
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            static_reals=['x','y','z', 'elevation'],  # Static features
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

    def train(
    self,
    training_data: TimeSeriesDataSet, 
    validation_data: TimeSeriesDataSet = None, 
    hidden_size=32, 
    learning_rate=0.03, 
    batch_size=32, 
    max_epochs=50
    ):
        self.train_loader = training_data.to_dataloader(train=True, batch_size=batch_size, num_workers=4)
        self.val_loader = validation_data.to_dataloader(train=False, batch_size=batch_size, num_workers=4)

        # Setup trainer
        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
        )
        lr_logger = LearningRateMonitor()  # log learning rate
        
        # Add ModelCheckpoint callback to save the best model
        from lightning.pytorch.callbacks import ModelCheckpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints/",
            filename="{epoch}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            verbose=True
        )
        
        #logger = TensorBoardLogger("lightning_logs")

        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            enable_model_summary=True,
            gradient_clip_val=0.1,
            limit_train_batches=50,  # comment in for training, running valiation every 30 batches
            #fast_dev_run=True,
            callbacks=[early_stop_callback, checkpoint_callback],  # Added checkpoint_callback here
            #logger=logger,
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
        with open("checkpoints/output_transformer.pkl", "wb") as f:
            pickle.dump(self.tft.output_transformer, f)

    def load_best_model(self, checkpoint_path=None):
        """
        Load the best model from a checkpoint
        Parameters:  checkpoint_path : str, optional
            Path to a specific checkpoint file. If None, attempts to find the best checkpoint
            from the trainer if training has occurred.
        Returns: The loaded model
        """
        import os
        checkpoint_path = "checkpoints/epoch=3-val_loss=0.06.ckpt"
        # if checkpoint_path is None:
        #     # Check if training has been performed
        #     if not hasattr(self, 'trainer'):
        #         raise ValueError("No trainer available. Either train the model first or provide a checkpoint_path.")
            
        #     # Find the checkpoint callback in the trainer's callbacks
        #     checkpoint_callback = None
        #     for callback in self.trainer.callbacks:
        #         if hasattr(callback, 'best_model_path'):
        #             checkpoint_callback = callback
        #             break
                    
        #     if checkpoint_callback is None or not checkpoint_callback.best_model_path:
        #         raise ValueError("No checkpoint found from training. Provide a specific checkpoint_path.")
                
        #     checkpoint_path = checkpoint_callback.best_model_path
        
        # Verify the checkpoint file exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        # Load the model
        best_tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
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

    def predict(self, location_data, prediction_length=14):
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
        # Load the best model
        best_tft = self.load_best_model()
        
        if prediction_length is None:
            prediction_length = self.max_prediction_length
        
        # Ensure we have enough history for encoder
        if len(location_data) < self.max_encoder_length:
            raise ValueError(f"Input data must contain at least {self.max_encoder_length} time steps for encoding")
        
        # Get the most recent data for encoding context
        encoder_data = location_data.tail(self.max_encoder_length).copy()
        
        # Create future timestamps for prediction
        last_timestamp = encoder_data['timestamp'].max()
        future_timestamps = [last_timestamp + i + 1 for i in range(prediction_length)]
        
        # Create decoder data frame for prediction
        decoder_data = pd.DataFrame({
            'timestamp': future_timestamps,
            'group_id': [encoder_data['group_id'].iloc[0]] * prediction_length,
            'x': [encoder_data['x'].iloc[0]] * prediction_length,
            'y': [encoder_data['y'].iloc[0]] * prediction_length,
            'z': [encoder_data['z'].iloc[0]] * prediction_length,
            'elevation': [encoder_data['elevation'].iloc[0]] * prediction_length
        })
        
        # Add required weather variables with placeholder values
        weather_cols = [
            "avg_temperature_2m", "min_temperature_2m", "max_temperature_2m",
            "dewfrostpoint_2m", "precipitation", "avg_windspeed_2m",
            "min_windspeed_2m", "max_windspeed_2m", "avg_windspeed_10m",
            "min_windspeed_10m", "max_windspeed_10m", "avg_windspeed_50m",
            "min_windspeed_50m", "max_windspeed_50m", "humidity_2m",
            "surface_pressure", "transpiration", "evaporation"
        ]
        
        # Add target variables with placeholder NaN values
        target_cols = ["num_deaths", "num_injured", "damage_cost", "disastertype"]
        
        for col in weather_cols + target_cols:
            if col == "disastertype":
                # Use the most common disaster type from the encoder data as placeholder
                decoder_data[col] = encoder_data[col].mode()[0]
            else:
                decoder_data[col] = 0 # 0 for the rest of the values
        
        # Combine encoder and decoder data
        prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)
        
        # Generate predictions
        raw_predictions = best_tft.predict(
            prediction_data,
            mode="raw",
            return_x=True,
            trainer_kwargs=dict(accelerator="cpu"),
        )
        print("raw_predictions")
        print(raw_predictions)
        # Process the results
        result = {}
        prediction_idx = slice(-prediction_length, None)    # For numerical targets, extract median predictions (0.5 quantile)
        output_dict = raw_predictions.output.prediction     # Get the output dictionary from raw predictions
        print(raw_predictions.output)
        print("prediction index")
        print(prediction_idx)
        print("output_dict")
        print(output_dict)
        # Extract predictions for each target
        deaths_pred = np.array(output_dict[0][prediction_idx])      # Deaths predictions with quantiles
        injured_pred = np.array(output_dict[1][prediction_idx])     # Injured predictions with quantiles
        damage_pred = np.array(output_dict[2][prediction_idx])      # Damage predictions with quantiles
        disaster_probs = output_dict[3][prediction_idx, :].softmax(dim=-1)
        print("disasterprobs")
        print(disaster_probs)

        # Get category mapping for disaster types
        if best_tft.output_transformer is None:
            with open("checkpoints/output_transformer.pkl", "rb") as f:
                best_tft.output_transformer = pickle.load(f)
        #disaster_categories = best_tft.output_transformer.transformation["disastertype"].categories_    later fix it
        disaster_categories = ['none','storm','flood','mass movement (dry)','extreme temperature ' ,'landslide']
        # Results for API consumption
        result["predictions"] = {
            "timestamps": future_timestamps,
            "deaths": {
                "median": deaths_pred[0,:, 1],  # 0.5 quantile
                "lower": deaths_pred[0,:, 0],   # Lower quantile
                "upper": deaths_pred[0,:, 2]    # Upper quantile
            },
            "injured": {
                "median": injured_pred[0,:, 1],
                "lower": injured_pred[0,:, 0],
                "upper": injured_pred[0,:, 2]
            },
            "damage_cost": {
                "median": damage_pred[0,:, 1],
                "lower": damage_pred[0,:, 0],
                "upper": damage_pred[0,:, 2]
            }
        }
        
        # Add disaster type predictions
        result["disaster_probabilities"] = {
            disaster_categories[i]: disaster_probs[:, i].tolist() 
            for i in range(len(disaster_categories))
        }
        predicted_disaster_types = []
        
        for i in range(len(future_timestamps)):
            for idx, prob in enumerate(disaster_probs):
                if isinstance(prob, np.ndarray):
                    prob = prob[0]  # Get the first value if it's an array
            if disaster_probs[0][i].size == 0:
                predicted_category = 'unknown'  # or some fallback category
            elif disaster_probs[0][i].shape[0] > 1:
                predicted_category_idx = np.argmax(disaster_probs[0][i])  # Get index of max probability
                if predicted_category_idx < len(disaster_categories):
                    predicted_category = disaster_categories[predicted_category_idx]
                    print("chosen")
                else:
                    predicted_category = 'unknown'  # Fallback if the index is out of range
            else:
                # If only one category is predicted, take it as the result
                predicted_category = disaster_categories[0][0]  # or some default category
            predicted_disaster_types.append(predicted_category)

        # Now, predicted_disaster_types will hold the correctly mapped categories
        result["predicted_disaster_types"] = predicted_disaster_types
        
        return result