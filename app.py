from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
from flask import render_template
from keras.models import load_model


import tensorflow as tf
import pandas as pd

# Load the trained model
model = tf.keras.models.load_model('horsepower_model.h5')

# Create a Flask application
app = Flask(__name__)
# Define the prediction endpoint

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json()

    # Preprocess the input data
    # You'll need to adapt this based on your dataset's preprocessing steps
    input_data = pd.DataFrame(data, index=[0])

    # Make predictions using the trained model
    predictions = model.predict(input_data)

    # Prepare the response
    response = {'predictions': predictions.tolist()}

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

