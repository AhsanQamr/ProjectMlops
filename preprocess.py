import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error

# Loading data from CSV file
sensor_data = pd.read_csv('data/dummy_sensor_data.csv')

# Convert 'Timestamp' column to datetime format
sensor_data['Timestamp'] = pd.to_datetime(sensor_data['Timestamp'])

# Extract 'Hour' from 'Timestamp' for potential use as a feature
sensor_data['Hour'] = sensor_data['Timestamp'].dt.hour

# Label encode 'Machine_ID' and 'Sensor_ID' columns for numeric representation
label_encoder_machine = LabelEncoder()
sensor_data['Machine_ID'] = label_encoder_machine.fit_transform(sensor_data['Machine_ID'])

label_encoder_sensor = LabelEncoder()
sensor_data['Sensor_ID'] = label_encoder_sensor.fit_transform(sensor_data['Sensor_ID'])

# Standardize numerical columns ('Hour', 'Machine_ID', 'Sensor_ID', 'Reading') using StandardScaler
data_scaler = StandardScaler()
sensor_data[['Hour', 'Machine_ID', 'Sensor_ID', 'Reading']] = data_scaler.fit_transform(sensor_data[['Hour', 'Machine_ID', 'Sensor_ID', 'Reading']])

sensor_data.to_csv("data/preprocessed_data.csv")
