import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

# Setting MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Loading data
data = pd.read_csv("data/preprocessed_data.csv")

# Define features (X) and target variable (y)
X = data[['Hour', 'Machine_ID', 'Sensor_ID']]
y = data['Reading']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Define lists of hyperparameters to tune
num_estimators_list = [60, 150, 200]
max_tree_depth_list = [10, 20, 30]

# Iterate over hyperparameter combinations
for num_estimators in num_estimators_list:
    for max_tree_depth in max_tree_depth_list:
        with mlflow.start_run():
            # Log hyperparameters with updated names
            mlflow.log_param("num_estimators", num_estimators)
            mlflow.log_param("max_tree_depth", max_tree_depth)

            # Build the RandomForestRegressor with the current hyperparameters
            rf_model = RandomForestRegressor(n_estimators=num_estimators, max_depth=max_tree_depth, random_state=42)

            # Fit the model
            rf_model.fit(X_train, y_train)

            # Make predictions
            predictions = rf_model.predict(X_test)

            # Log metrics
            mse = mean_squared_error(y_test, predictions)
            mlflow.log_metric("mse", mse)

            # Log the model
            mlflow.sklearn.log_model(rf_model, "random_forest_model_{n}_{m}".format(n=num_estimators, m=max_tree_depth))
