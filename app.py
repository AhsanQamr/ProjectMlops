from flask import Flask, render_template, request, flash
import mlflow.pyfunc
import pandas as pd
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask App Initialization
flask_app = Flask(__name__)
flask_app.secret_key = 'your_secret_key'  # Replace with your secret key

# Configuration
MODEL_PATH = "model_selection/"  # Externalize configuration
HOST = '0.0.0.0'
PORT = 5001

# Load MLflow Model
try:
    model = mlflow.pyfunc.load_model(MODEL_PATH)
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# Route for File Upload and Prediction
@flask_app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        uploaded_csv = request.files['file']

        if uploaded_csv and uploaded_csv.filename != '':
            if not uploaded_csv.filename.endswith('.csv'):
                flash('File format not supported. Please upload a CSV file.')
                return render_template('index.html')

            try:
                # Convert the uploaded CSV file to a DataFrame
                input_data = pd.read_csv(uploaded_csv)

                # Select relevant features for the model
                features = input_data[['Hour', 'Machine_ID', 'Sensor_ID']]

                # Generate predictions using the model
                model_predictions = model.predict(features)

                # Prepare the results for display
                prediction_results = pd.DataFrame({'Prediction': model_predictions})
                return render_template('prediction.html', prediction_results=prediction_results.to_html())
            except Exception as e:
                logger.error(f"Error processing file: {e}")
                flash('Error processing file. Please try again.')
                return render_template('index.html')
        else:
            flash('No file selected. Please upload a CSV file.')
    
    return render_template('index.html')

# Run the Flask Application
if __name__ == '__main__':
    flask_app.run(host=HOST, port=PORT)
