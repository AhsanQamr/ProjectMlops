# import mlflow.sklearn
# from mlflow.tracking import MlflowClient
# from sklearn.metrics import mean_squared_error
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor

# # Set MLflow tracking URI
# mlflow.set_tracking_uri("http://127.0.0.1:5003")

# # Load data
# preprocessed_data = pd.read_csv("data\preprocessed_data.csv")


# # Define features (X) and target variable (y)
# features = preprocessed_data[['Hour', 'Machine_ID', 'Sensor_ID']]
# target = preprocessed_data['Reading']

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, shuffle=False)

# # Retrieve the best run using MLflow
# mlflow_client = MlflowClient()
# experiment_name = "Default"  # Replace with your actual experiment name
# experiment = mlflow_client.get_experiment_by_name(experiment_name)

# if experiment:
#     experiment_id = experiment.experiment_id
# else:
#     print(f"Experiment '{experiment_name}' not found.")
#     exit()

# # Search for runs in the experiment
# runs = mlflow_client.search_runs(experiment_ids=[experiment_id], filter_string="", order_by=["metrics.mse ASC"])

# # Iterate through runs to find the best one
# best_run = None
# best_mse = float('inf')  # Initialize with a large value

# for run in runs:
#     mse = run.data.metrics.get("mse")
#     if mse is not None and mse < best_mse:
#         best_mse = mse
#         best_run = run

# # Check if any runs were found
# if best_run is not None:
#     print(f"Best Run ID: {best_run.info.run_id}")
#     print(f"Best MSE: {best_mse}")

#     # Load the best model
#     best_model = mlflow.sklearn.load_model("runs:/" + best_run.info.run_id + "/random_forest_model")

#     # Register the best model in the Model Registry
#     mlflow.register_model("runs:/" + best_run.info.run_id + "/random_forest_model", "MLOPSPROJECTMODEL")

#     # Save the loaded model to a file or deploy it as needed
#     # For example, if you want to save it as a .pkl file:
#     mlflow.sklearn.save_model(best_model, "model_selection")

# else:
#     print("No runs found in the experiment.")


import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import requests

# Function to check if MLflow server is up and running
def is_server_running(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.ConnectionError:
        return False

# MLflow server URL
mlflow_server_url = "http://127.0.0.1:5003"

# Check if MLflow server is running
if not is_server_running(mlflow_server_url):
    print(f"MLflow server at {mlflow_server_url} is not running. Please start the server and try again.")
    exit()

# Set MLflow tracking URI
mlflow.set_tracking_uri(mlflow_server_url)

# Load data
preprocessed_data = pd.read_csv("data/preprocessed_data.csv")

# Define features (X) and target variable (y)
features = preprocessed_data[['Hour', 'Machine_ID', 'Sensor_ID']]
target = preprocessed_data['Reading']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, shuffle=False)

# Retrieve the best run using MLflow
mlflow_client = MlflowClient()
experiment_name = "Default"  # Replace with your actual experiment name
experiment = mlflow_client.get_experiment_by_name(experiment_name)

if experiment:
    experiment_id = experiment.experiment_id
else:
    print(f"Experiment '{experiment_name}' not found.")
    exit()

# Search for runs in the experiment
runs = mlflow_client.search_runs(experiment_ids=[experiment_id], filter_string="", order_by=["metrics.mse ASC"])

# Iterate through runs to find the best one
best_run = None
best_mse = float('inf')  # Initialize with a large value

for run in runs:
    mse = run.data.metrics.get("mse")
    if mse is not None and mse < best_mse:
        best_mse = mse
        best_run = run

# Check if any runs were found
if best_run is not None:
    print(f"Best Run ID: {best_run.info.run_id}")
    print(f"Best MSE: {best_mse}")

    # Attempt to load the best model
    try:
        best_model = mlflow.sklearn.load_model("runs:/" + best_run.info.run_id + "/random_forest_model")
        
        # Register the best model in the Model Registry
        mlflow.register_model("runs:/" + best_run.info.run_id + "/random_forest_model", "temp_model")

        # Save the loaded model to a file or deploy it as needed
        # For example, if you want to save it as a .pkl file:
        mlflow.sklearn.save_model(best_model, "model_selection")
    except Exception as e:
        print(f"Failed to load model: {e}")

else:
    print("No runs found in the experiment.")