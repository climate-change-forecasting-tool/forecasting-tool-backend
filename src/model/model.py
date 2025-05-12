from typing import List
import h3
import pandas as pd
import numpy as np
from pytorch_forecasting import NaNLabelEncoder, TemporalFusionTransformer, TimeSeriesDataSet, GroupNormalizer, QuantileLoss
from sklearn.preprocessing import StandardScaler
import lightning.pytorch as pl
import torch
from src.configuration.config import Config
from datetime import timedelta


import logging
logging.basicConfig(level=logging.INFO)
import warnings
warnings.filterwarnings("ignore") 

from src.utility import latlon_to_xyz, get_astronomical_season

class TFTransformer: 
    def __init__(self):
        df = pd.read_parquet(path=Config.summary_dataset_filepath, engine="pyarrow")

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

        # logging.info("Means")
        # logging.info(self.data_means)
        # logging.info("Vars")
        # logging.info(self.data_stds)

        self.train_df[self.unknown_climate_reals] = scaler.transform(self.train_df[self.unknown_climate_reals])
        self.val_df[self.unknown_climate_reals] = scaler.transform(self.val_df[self.unknown_climate_reals])
        self.test_df[self.unknown_climate_reals] = scaler.transform(self.test_df[self.unknown_climate_reals])

        # Define max prediction & history length
        self.max_prediction_length = 14   # Forecast next 'x' days; 14
        self.max_encoder_length = 365     # Use past 'x' days for prediction; 365

        self.tft = self.load_best_model()

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

    def load_best_model(self):
        """
        Save on machine with CUDA GPU with the following:

        def model_to_cpu(self, checkpoint_path):
            checkpoint_model = TemporalFusionTransformer.load_from_checkpoint(
                checkpoint_path=checkpoint_path, 
                map_location=torch.device("cpu")
            )

            new_checkpoint_path = "checkpoints/model.ckpt"
            torch.save(checkpoint_model, new_checkpoint_path)

        """

        checkpoint_path = Config.tft_checkpoint_path
        
        logging.info(f"Using best model from: {checkpoint_path}")

        best_tft: TemporalFusionTransformer = torch.load(
            checkpoint_path, 
            map_location=torch.device("cpu"),
            weights_only=False
        )
        best_tft.eval()

        return best_tft

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
        
        raw_predictions = self.tft.predict(
            dataloader, # or tsds
            mode="raw",
            return_x=True, 
            trainer_kwargs=dict(accelerator=Config.tft_accelerator)
        )

        raw_output = raw_predictions.output.prediction

        predictions = {}
    
        # Extract and denormalize continuous targets
        for idx, target in enumerate([self.target]):
            denormalized_pred = np.array(raw_output[idx].cpu())

            denormalized_pred = np.sort(denormalized_pred, axis=1)

            median_idx = denormalized_pred.shape[1] // 2
            
            predictions[target + '_lower'] = denormalized_pred[:, 0]
            predictions[target + '_median'] = denormalized_pred[:, median_idx]
            predictions[target + '_upper'] = denormalized_pred[:, -1]

        predictions_df = pd.DataFrame(predictions)
        
        return predictions_df
        
    def predict_test(self):
        # group_id = self.train_df.iloc[0]['group_id']

        # prev_df = pd.concat([self.train_df, self.val_df])
        # prev_df = prev_df[prev_df['group_id'] == group_id]
        # results = self.predict(location_data=prev_df)

        # logging.info("results:")
        # logging.info(results)

        # analysis_df = self.test_df.copy()
        # analysis_df = analysis_df[analysis_df['group_id'] == group_id]
        # df_to_analyze = self.test_df.head(self.max_prediction_length)

        # results = self.predict_and_digest(longitude=38.027955, latitude=79.251775)
        results = self.predict_and_digest(longitude=-74.006, latitude=40.7128)

        logging.info(results)

        return results
        
    
    def predict_and_digest(self, longitude: float, latitude: float):
        """
        Call this function for the routing point model query
        """
        try:
            group_id = h3.str_to_int(
                h3.latlng_to_cell(
                    lat=latitude, 
                    lng=longitude, 
                    res=Config.hexagon_resolution
                )
            ) + 1

            location_df = pd.concat([self.train_df, self.val_df, self.test_df])
            location_df = location_df[location_df['group_id'] == group_id]

            if len(location_df) < self.max_encoder_length:
                raise Exception("Bad group id.")

            prediction_df = self.predict(location_data=location_df)

            # convert Kelvin to Fahrenheit
            digested_df = (prediction_df - 273.15) * 9./5. + 32.

            digested_dict = dict()

            for index, row in digested_df.iterrows():
                digested_dict.update({index: row['t2m_mean_median']})

            return digested_dict
        except Exception as e:
            raise e


