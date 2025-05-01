import csv
from datetime import datetime

from .services import NOAAService, NASAService
from .configuration.config import Config
import os
from dotenv import load_dotenv
import logging

from .dataset_summary import SummaryDataset
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

load_dotenv('.env')
"""
EM-DAT: disaster data
    - https://doc.emdat.be/docs/legal/terms-of-use/
NASA Earthdata | GDIS: geocoded disaster data
NASA POWER API: weather data

"""

sds = SummaryDataset()

if Config.generate_summary_dataset:
    if Config.recreate_summary_dataset:
        sds.clear_dataset()

    sds.upload_data(start_date=Config.start_date, end_date=Config.end_date)

if Config.activate_tft:
    from .model import TFTransformer
    import pandas as pd
    import numpy as np
    from sklearn.discriminant_analysis import StandardScaler

    tft = TFTransformer(parquet_df=Config.summary_dataset_filepath)
    all_df = pd.read_parquet(Config.summary_dataset_filepath)
    print("dataset loading completed")
    # 2a. Select the subset for prediction
    #subset_df = df[df["latitude"] > 60]
    df = all_df
    row1 = df.iloc[0]
    long1 = row1['longitude']
    lat1 = row1['latitude']
    #one point
    df = df[(df['longitude'] == long1) & (df['latitude'] == lat1)]
    #process data
    df['disastertype'] = df['disastertype'].astype('category')
    df['timestamp'] = df['timestamp'].astype(int)
    scaler = StandardScaler()
    df[['elevation', 'num_deaths', 'num_injured', 'damage_cost']] = scaler.fit_transform(df[['elevation', 'num_deaths', 'num_injured', 'damage_cost']])
    def latlon_to_xyz(lat, lon):
            lat_rad = np.radians(lat)
            lon_rad = np.radians(lon)
            x = np.cos(lat_rad) * np.cos(lon_rad)
            y = np.cos(lat_rad) * np.sin(lon_rad)
            z = np.sin(lat_rad)
            return x, y, z
    df['x'], df['y'], df['z'] = latlon_to_xyz(df['latitude'], df['longitude'])
    max_datapoints = df['timestamp'].max()

    print("datapoints")
    print(max_datapoints)
    
    df['group_id'] = df.groupby(['longitude', 'latitude']).ngroup()
    df.sort_values(by = ['timestamp', 'group_id'])
    df.drop(['longitude', 'latitude'], axis=1, inplace=True)
    df["disastertype"] = df["disastertype"].fillna('none')
    #
    
    #-------------------------
    predictions = tft.predict(location_data=df,prediction_length=14)
    print(predictions)
    timestamps = predictions['predictions']['timestamps'][:14]
    deaths_median = predictions['predictions']['deaths']['median']
    deaths_lower = predictions['predictions']['deaths']['lower'][0]
    deaths_upper = predictions['predictions']['deaths']['upper'][0]

    injured_median = predictions['predictions']['injured']['median']
    injured_lower = predictions['predictions']['injured']['lower'][0]
    injured_upper = predictions['predictions']['injured']['upper'][0]

    damage_median = predictions['predictions']['damage_cost']['median']
    damage_lower = predictions['predictions']['damage_cost']['lower'][0]
    damage_upper = predictions['predictions']['damage_cost']['upper'][0]
    
    # print(timestamps)
    # print(deaths_median)
    # print(injured_median)
    # print(damage_median)
    # Build a proper table
    predictions_df = pd.DataFrame({
        'timestamp': timestamps,
        'deaths_median': deaths_median,
        'deaths_lower': deaths_lower,
        'deaths_upper': deaths_upper,
        'injured_median': injured_median,
        'injured_lower': injured_lower,
        'injured_upper': injured_upper,
        'damage_cost_median': damage_median,
        'damage_cost_lower': damage_lower,
        'damage_cost_upper': damage_upper,
    })

    print(predictions_df)

    disaster_probs = predictions['disaster_probabilities']

    # Flatten into a list of dicts
    disaster_probs_list = []
    for disaster_type, probs in disaster_probs.items():
        # probs is [[none_prob, actual_prob]]
        none_prob, disaster_prob = probs[0]
        disaster_probs_list.append({
            'disaster_type': disaster_type,
            'none_probability': none_prob,
            'disaster_probability': disaster_prob,
        })

    # Build a DataFrame
    disaster_probs_df = pd.DataFrame(disaster_probs_list)
    print(disaster_probs_df)
    predictions_df.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv")

