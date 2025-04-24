

# Import necessary libraries
import os
import requests
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
import plotly.graph_objects as go
import streamlit as st
import requests_html
from requests_html import HTMLSession
from datetime import datetime, timedelta
import pandas as pd
import plotly

#from google.colab import drive
#drive.mount('/content/drive')
#model_path = '/content/drive/My Drive/models/my_trained_model_water_level_Sogndalsvatn_April2024.keras'
#model = tf.keras.models.load_model(model_path)


# Check if everything works:
print("Packages loaded successfully!")
# Constants
API_KEY = os.environ.get('NVE_API_KEY', '3G2Xx1Sw8kGpYdbzdskSew==')
BASE_URL = 'https://hydapi.nve.no/api/v1/Observations'
SEQUENCE_LENGTH = 24
FORECAST_HORIZON = 24

# Streamlit Inputs
st.title("Water Level Prediction for Sogndalsvatn")
st.sidebar.header("User Inputs")
station_id = st.sidebar.text_input("Station ID", "77.3.0")
parameter = st.sidebar.text_input("Parameter", "1000")
forecast_days = st.sidebar.slider("Forecast Days", 1, 7, 3)
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=7))
end_date = st.sidebar.date_input("End Date", datetime.now() + timedelta(days=forecast_days))

# Load the trained GRU model
#@st.cache_resource
#def load_model():
#    return tf.keras.models.load_model(r"C:\Users\mwa\models\my_trained_model_water_level_Sogndalsvatn_April2024.keras")

import streamlit as st
#from google.colab import files
import tensorflow as tf

# Download the model file to the current working directory
#files.download('/content/drive/My Drive/models/my_GRU_model_waterlevel.keras')

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('my_GRU_model_waterlevel.keras') # Load from the current directory
    return model

# Correct way with raw string
#model = tf.keras.models.load_model(r"C:\Users\mwa\models\my_trained_model_water_level_Sogndalsvatn_April2024.keras")


# Function to fetch weather data
@st.cache_data
def fetch_weather_data(start_date, end_date):
    xcoord = [70022, 67095, 62755, 70175]
    ycoord = [6825600, 6835892, 6828763, 6830460]
    attributes = ['rr3h', 'tm3h']
    ind = pd.date_range(start=start_date, end=end_date, freq='3h')
    df_weather = pd.DataFrame(index=ind)
    session = HTMLSession()

    for m, coord in enumerate(zip(xcoord, ycoord), start=1):
        for atr in attributes:
            url = f'http://gts.nve.no/api/GridTimeSeries/{coord[0]}/{coord[1]}/{start_date}/{end_date}/{atr}.json'
            r = session.get(url)
            if r.status_code == 200:
                df = pd.DataFrame(r.json(), index=ind)
                df = df.drop(columns=['Theme', 'FullName', 'NoDataValue', 'X', 'Y', 'StartDate', 'EndDate',
                                      'PrognoseStartDate', 'Unit', 'TimeResolution', 'Altitude'])
                df = df.rename(columns={'Data': r.json()['Theme'] + str(m)})
                df_weather = pd.concat([df_weather, df], axis=1)
            else:
                st.error(f"Error fetching data for {atr} at coordinates {coord}: Status code {r.status_code}")
    return df_weather

# Function to fetch inflow data
@st.cache_data
def fetch_inflow_data(station_id, parameter, start_date, end_date):
    headers = {'X-API-Key': API_KEY, 'accept': 'application/json'}
    params = {
        'StationId': station_id,
        'Parameter': parameter,
        'ResolutionTime': '60',
        'ReferenceTime': f'{start_date}/{end_date}'
    }
    response = requests.get(BASE_URL, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        if data and 'data' in data and len(data['data']) > 0:
            series_data = data['data'][0]
            observations = series_data['observations']
            df = pd.DataFrame(observations)
            df['Date'] = pd.to_datetime(df['time'])
            df = df[['Date', 'value']].set_index('Date')
            df = df.resample('3h').mean()
            df.index = df.index.tz_localize(None)
            return df.rename(columns={'value': 'inflow'})
    st.error("Failed to fetch inflow data.")
    return None

# Function to preprocess data
def preprocess_data(weather_data, inflow_data):
    dataset = weather_data.join(inflow_data)
    dataset = dataset.mask(dataset > 1000)
    for column in dataset.columns:
        if column != 'inflow':
            dataset[column] = (dataset[column] - dataset[column].min()) / (dataset[column].max() - dataset[column].min())
            dataset = dataset.interpolate(method='linear', limit_direction='both')

    return dataset

# Function to prepare sequences
def prepare_sequences(dataset):
    X, y = [], []
    for i in range(len(dataset) - SEQUENCE_LENGTH - FORECAST_HORIZON):
        X_seq = dataset.iloc[i:(i + SEQUENCE_LENGTH)].values
        forecast_values = dataset.iloc[i + SEQUENCE_LENGTH:i + SEQUENCE_LENGTH + FORECAST_HORIZON].values
        forecast_values[:, dataset.columns.get_loc('inflow')] = 0
        X_seq_with_forecast = np.concatenate([X_seq, forecast_values])
        X.append(X_seq_with_forecast)
        y.append(dataset['inflow'].iloc[i + SEQUENCE_LENGTH:i + SEQUENCE_LENGTH + FORECAST_HORIZON].values)
    return np.array(X), np.array(y)

# Function to plot predictions
def plot_predictions(dataset, y_pred):
    future_date_range = pd.date_range(end=dataset.index[-1], periods=FORECAST_HORIZON + 1, freq='3h')[1:]
    plot_df = pd.DataFrame({'Predicted': y_pred[-1]}, index=future_date_range)
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=dataset.index[:-FORECAST_HORIZON], y=dataset['inflow'], mode='lines', name='Past Water Level', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Predicted'], mode='lines', name='Predicted', line=dict(color='red', dash='dash')))
    fig.update_layout(title='Water Level Prediction for Sogndalsvatn', xaxis_title='Date', yaxis_title='Water Level')
    return fig

    return fig

# Main Streamlit App
st.header("Fetching Data")
weather_data = fetch_weather_data(start_date.strftime('%d.%m.%Y'), end_date.strftime('%d.%m.%Y'))
inflow_data = fetch_inflow_data(station_id, parameter, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

model = load_model()
if inflow_data is not None:
    st.success("Data fetched successfully!")

    st.header("Preprocessing Data")
    dataset = preprocess_data(weather_data, inflow_data)
    st.write("Data preprocessing completed!")

    st.header("Making Predictions")
    X, y = prepare_sequences(dataset)
    y_pred = model.predict(X)
    st.success("Predictions completed!")

    st.header("Prediction Results")
    fig = plot_predictions(dataset, y_pred)
    st.plotly_chart(fig)
else:
    st.error("Failed to fetch inflow data.")
