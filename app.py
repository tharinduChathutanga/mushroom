# Import necessary libraries
from flask import Flask, request, jsonify
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from flask_cors import CORS

# Create your Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Step 1: Data Collection
# Read data from a CSV file named 'mymushroom.csv'
data = pd.read_csv('mymushroom.csv')

# Step 2: Data Preprocessing
# Define a function to filter data based on mushroom type
def filter_data(df, mushroom_type):
    return df[df['Type'] == mushroom_type].reset_index(drop=True)

# Define a function to fit an ARIMA model to the filtered data
def fit_arima(data):
    model = ARIMA(data['Price'].values, order=(1, 0, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=1)[0]
    return forecast

# Define a function to preprocess data for the RNN model
def preprocess_rnn(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Price']])
    return scaled_data, scaler

# Define a function to create an RNN model
def create_rnn_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    req_data = request.json
    mushroom_type = req_data['mushroom_type']
    harvest_amount = req_data['harvest_amount']
    cost = req_data['cost']

    # Filter the data based on mushroom type
    data_filtered = filter_data(data, mushroom_type)

    # Perform ARIMA prediction
    arima_prediction = fit_arima(data_filtered)
    profit_arima = (arima_prediction * harvest_amount) - (cost * harvest_amount)

    # Preprocess data for the RNN model
    scaled_data, scaler = preprocess_rnn(data_filtered)
    X = scaled_data[:-1]
    y = scaled_data[1:]

    # Create and train the RNN model
    rnn_model = create_rnn_model(input_shape=(X.shape[1], 1))
    history = rnn_model.fit(X, y, epochs=50, batch_size=16, verbose=0)
    
    # Make RNN prediction
    rnn_prediction = rnn_model.predict(X[-1].reshape(1, X.shape[1], 1))
    rnn_prediction = scaler.inverse_transform(rnn_prediction)[0][0]
    profit_rnn = (rnn_prediction * harvest_amount) - (cost * harvest_amount)

    # Prepare the result as a dictionary
    result = {
        'mushroom_type': mushroom_type,
        'harvest_amount': '{:.2f}'.format(harvest_amount),
        'cost': '{:.2f}'.format(cost),
        'arima_prediction': '{:.2f}'.format(arima_prediction.tolist()),
        'profit_arima': '{:.2f}'.format(profit_arima.tolist()),
        'rnn_prediction': '{:.2f}'.format(rnn_prediction.tolist()),
        'profit_rnn': '{:.2f}'.format(profit_rnn.tolist())
    }
    
    # Return the result as JSON
    return jsonify(result)

# Start the Flask app if this script is executed
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
