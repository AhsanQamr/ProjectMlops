from flask import Flask, render_template, request
import mlflow.pyfunc
import pandas as pd

# Initializing Flask app
flask_app = Flask(__name__)

# Path to the MLflow model
model_path = "model_selection/"
# Load the MLflow model
model = mlflow.pyfunc.load_model(model_path)

# Route for handling file uploads and predictions


@flask_app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        # Retrieve the uploaded file
        uploaded_csv = request.files['file']

        if uploaded_csv.filename != '':
            # Convert the uploaded CSV file to a DataFrame
            input_data = pd.read_csv(uploaded_csv)

            # Select relevant features for the model
            features = input_data[['Hour', 'Machine_ID', 'Sensor_ID']]

            # Generate predictions using the model
            model_predictions = model.predict(features)

            # Prepare the results for display

            # hello world


            prediction_results = pd.DataFrame({'Prediction': model_predictions})
            return render_template('prediction.html', prediction_results=prediction_results.to_html())
    

    return render_template('index.html')


# Run the Flask application
if __name__ == '__main__':
    flask_app.run(host='0.0.0.0', port=5001)
