from multiprocessing import freeze_support
# from flask import Flask, request, jsonify
import pickle
import numpy as np
# from flask_cors import CORS
import pandas as pd
from src.configuration.config import Config
from src.services.nasa_power_service import NASAService

# app = Flask(__name__)
# CORS(app)
# with open("checkpoints/output_transformer.pkl", "rb") as f:
#     model = pickle.load(f)

# nasa_service = NASAService()


# @app.route("/", methods=["POST"])
# def predict():
#     prediction_length=Config.tft_prediction_length
#     try:
#         data = request.get_json()
#         longitude = float(data.get("longtitude"))
#         latitude = float(data.get("latitude"))
#     except Exception as e:
#         return jsonify({'ok': False, 'errors': e}), 400
    
#     def latlon_to_xyz(lat, lon):
#         lat_rad = np.radians(lat)
#         lon_rad = np.radians(lon)
#         x = np.cos(lat_rad) * np.cos(lon_rad)
#         y = np.cos(lat_rad) * np.sin(lon_rad)
#         z = np.sin(lat_rad)
#         return x, y, z
    
#     x, y, z = latlon_to_xyz(latitude, longitude)


    
    
#     encoder_data = location_data.tail(self.max_encoder_length).copy()
    
#     # Create future timestamps for prediction
#     last_timestamp = encoder_data['timestamp'].max()
#     future_timestamps = [last_timestamp + i + 1 for i in range(prediction_length)]
    
#     # Create decoder data frame for prediction
#     decoder_data = pd.DataFrame({
#         'timestamp': future_timestamps,
#         'group_id': [encoder_data['group_id'].iloc[0]] * prediction_length,
#         'x': [x] * prediction_length,
#         'y': [y] * prediction_length,
#         'z': [z] * prediction_length,
#         'elevation': [encoder_data['elevation'].iloc[0]] * prediction_length
#     })
    
#     # Add required weather variables with placeholder values
#     weather_cols = [
#         "avg_temperature_2m", "min_temperature_2m", "max_temperature_2m",
#         "dewfrostpoint_2m", "precipitation", "avg_windspeed_2m",
#         "min_windspeed_2m", "max_windspeed_2m", "avg_windspeed_10m",
#         "min_windspeed_10m", "max_windspeed_10m", "avg_windspeed_50m",
#         "min_windspeed_50m", "max_windspeed_50m", "humidity_2m",
#         "surface_pressure", "transpiration", "evaporation"
#     ]
    
#     # Add target variables with placeholder NaN values
#     target_cols = ["num_deaths", "num_injured", "damage_cost", "disastertype"]
    
#     for col in weather_cols + target_cols:
#         if col == "disastertype":
#             # Use the most common disaster type from the encoder data as placeholder
#             decoder_data[col] = encoder_data[col].mode()[0]
#         else:
#             decoder_data[col] = 0 # 0 for the rest of the values
    
#     # Combine encoder and decoder data
#     prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)
    
#     # Generate predictions
#     raw_predictions = model.predict(
#         prediction_data,
#         mode="raw",
#         return_x=True,
#         trainer_kwargs=dict(accelerator="cpu"),
#     )
#     print("raw_predictions")
#     print(raw_predictions)
#     # Process the results
#     result = {}
#     prediction_idx = slice(-prediction_length, None)    # For numerical targets, extract median predictions (0.5 quantile)
#     output_dict = raw_predictions.output.prediction     # Get the output dictionary from raw predictions
#     print(raw_predictions.output)
#     print("prediction index")
#     print(prediction_idx)
#     print("output_dict")
#     print(output_dict)
#     # Extract predictions for each target
#     deaths_pred = np.array(output_dict[0][prediction_idx])      # Deaths predictions with quantiles
#     injured_pred = np.array(output_dict[1][prediction_idx])     # Injured predictions with quantiles
#     damage_pred = np.array(output_dict[2][prediction_idx])      # Damage predictions with quantiles
#     disaster_probs = output_dict[3][prediction_idx, :].softmax(dim=-1)
#     print("disasterprobs")
#     print(disaster_probs)

#     # Get category mapping for disaster types
#     if model.output_transformer is None:
#         with open("checkpoints/output_transformer.pkl", "rb") as f:
#             model.output_transformer = pickle.load(f)
#     #disaster_categories = model.output_transformer.transformation["disastertype"].categories_    later fix it
#     disaster_categories = ['none','storm','flood','mass movement (dry)','extreme temperature ' ,'landslide']
#     # Results for API consumption
#     result["predictions"] = {
#         "timestamps": future_timestamps,
#         "deaths": {
#             "median": deaths_pred[0,:, 1],  # 0.5 quantile
#             "lower": deaths_pred[0,:, 0],   # Lower quantile
#             "upper": deaths_pred[0,:, 2]    # Upper quantile
#         },
#         "injured": {
#             "median": injured_pred[0,:, 1],
#             "lower": injured_pred[0,:, 0],
#             "upper": injured_pred[0,:, 2]
#         },
#         "damage_cost": {
#             "median": damage_pred[0,:, 1],
#             "lower": damage_pred[0,:, 0],
#             "upper": damage_pred[0,:, 2]
#         }
#     }
    
#     # Add disaster type predictions
#     result["disaster_probabilities"] = {
#         disaster_categories[i]: disaster_probs[:, i].tolist() 
#         for i in range(len(disaster_categories))
#     }
#     predicted_disaster_types = []
    
#     for i in range(len(future_timestamps)):
#         for idx, prob in enumerate(disaster_probs):
#             if isinstance(prob, np.ndarray):
#                 prob = prob[0]  # Get the first value if it's an array
#         if disaster_probs[0][i].size == 0:
#             predicted_category = 'unknown'  # or some fallback category
#         elif disaster_probs[0][i].shape[0] > 1:
#             predicted_category_idx = np.argmax(disaster_probs[0][i])  # Get index of max probability
#             if predicted_category_idx < len(disaster_categories):
#                 predicted_category = disaster_categories[predicted_category_idx]
#                 print("chosen")
#             else:
#                 predicted_category = 'unknown'  # Fallback if the index is out of range
#         else:
#             # If only one category is predicted, take it as the result
#             predicted_category = disaster_categories[0][0]  # or some default category
#         predicted_disaster_types.append(predicted_category)

#     # Now, predicted_disaster_types will hold the correctly mapped categories
#     result["predicted_disaster_types"] = predicted_disaster_types
    
#     return result

if __name__ == '__main__':
    import src.main as main

    freeze_support()



"""

Temporal extent:
gdis: (1960-01-01)-(2018-12-31)
NASA climate data: (1980-12-31)-(?)


Spatial extent:
gdis: 90 N, -58 S, 180 E, -180 W



Data processing
1. Generate points around the globe to query (including a bounding box radius)
2. Use some timespan: start date to end date
3. For each point at some time, retrieve:
    a. timestamp
    b. longitude
    c. latitude
    d. elevation                    NASA POWER API
    e. disastertype                 EM-DAT / GDIS
    f. number of deaths             EM-DAT
    g. number of injured           EM-DAT
    h. property damage cost         EM-DAT
    i. ***weather data
        - temperature               NASA POWER API
        - precipitation             NASA POWER API
        - humidity                  NASA POWER API
        - air pressure              NASA POWER API
        - wind speed                NASA POWER API
        - wind direction            NASA POWER API
        - snow cover
    j. ***climate-change data (pollutants)
        - 
    k. ***urban data
        - population density
        - 
4. Save to dataset



Climate parameters
Mess around with hexagon resolution
Get population data

"""