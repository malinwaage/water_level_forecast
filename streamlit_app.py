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
st.title("Water Level/Discharge Prediction for Sogndalsvatn") # Updated title

# Add description
st.write("""
This application predicts water levels and inflow for Sogndalsvatn using a GRU-based deep learning model, 
which outperformed other methods. It uses temperature, precipitation, and historical data for forecasts 
several days in advance on a three-hour basis. Tests showed 99%-92% accuracy for water level predictions 
and 91%-78% accuracy for discharge predictions. Data is collected from NVE's open APIs.
""")
st.sidebar.header("User Inputs")
station_id = st.sidebar.text_input("Station ID", "77.3.0")
parameter = st.sidebar.selectbox("Parameter (1001:inflow/discharge, 1000:water-level)", ["1000", "1001"], index=0)  # Selectbox for parameter
forecast_days = st.sidebar.slider("Forecast Days", 1, 2, 3)
today = datetime.now()  
#start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=7))
#end_date = st.sidebar.date_input("End Date", datetime.now() + timedelta(days=forecast_days))
start_date = st.sidebar.date_input("Start Date", (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'))
#end_date = st.sidebar.date_input("End Date", (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d'))
end_date = st.sidebar.date_input("End Date", (datetime.now() + timedelta(forecast_days)).strftime('%Y-%m-%d'))

#.strftime('%d.%m.%Y')
# Load the trained model based on selected parameter
#@st.cache_resource
def load_model(parameter): 
    if parameter == "1000":
        model = tf.keras.models.load_model('my_GRU_model_waterlevel.keras')
    elif parameter == "1001":
        model = tf.keras.models.load_model('my_GRU_model_discharge3.keras')
    return model


# Load the trained GRU model
#@st.cache_resource
#def load_model():
#    return tf.keras.models.load_model(r"C:\Users\mwa\models\my_trained_model_water_level_Sogndalsvatn_April2024.keras")

import streamlit as st
#from google.colab import files
import tensorflow as tf

# Download the model file to the current working directory
#files.download('/content/drive/My Drive/models/my_GRU_model_waterlevel.keras')


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
            # Rename the column based on the parameter:
            if parameter == "1000":
                return df.rename(columns={'value': 'waterlevel'})
            elif parameter == "1001":
                return df.rename(columns={'value': 'discharge'})
    st.error("Failed to fetch inflow data.")
    return none
# Function to preprocess data

def preprocess_data(weather_data, inflow_data, parameter):
    min_max_values = {}
    dataset = weather_data.join(inflow_data)
    inflow_column = 'waterlevel' if parameter == "1000" else 'discharge'
    dataset = dataset.rename(columns={'value': inflow_column})
    dataset = dataset.mask(dataset > 1000)

    for column in dataset.columns:
        if column != inflow_column:
            dataset[column] = (dataset[column] - dataset[column].min()) / (dataset[column].max() - dataset[column].min())
            dataset = dataset.interpolate(method='linear', limit_direction='both')
    return dataset
    

# Function to prepare sequences
def prepare_sequences(dataset, parameter):  # Add 'parameter' argument
    X, y = [], []
    # Get the correct column name
    inflow_column = 'waterlevel' if parameter == "1000" else 'discharge' 

    for i in range(len(dataset) - SEQUENCE_LENGTH - FORECAST_HORIZON):
        X_seq = dataset.iloc[i:(i + SEQUENCE_LENGTH)].values
        forecast_values = dataset.iloc[i + SEQUENCE_LENGTH:i + SEQUENCE_LENGTH + FORECAST_HORIZON].values
        
        # Use inflow_column instead of 'inflow'
        forecast_values[:, dataset.columns.get_loc(inflow_column)] = 0  
        
        X_seq_with_forecast = np.concatenate([X_seq, forecast_values])
        X.append(X_seq_with_forecast)
        
        # Use inflow_column instead of 'inflow'
        y.append(dataset[inflow_column].iloc[i + SEQUENCE_LENGTH:i + SEQUENCE_LENGTH + FORECAST_HORIZON].values)  
    return np.array(X), np.array(y)
# Function to plot predictions

    
#old
def plot_predictions(dataset, y_pred, parameter):  # Add parameter argument
    future_date_range = pd.date_range(end=dataset.index[-1], periods=FORECAST_HORIZON + 1, freq='3h')[1:]
    plot_df = pd.DataFrame({'Predicted': y_pred[-1]}, index=future_date_range)
    fig = go.Figure()

    # Use 'waterlevel' or 'discharge' based on parameter
    data_column = 'waterlevel' if parameter == "1000" else 'discharge'  
    fig.add_trace(go.Scatter(x=dataset.index[:-FORECAST_HORIZON], y=dataset[data_column], mode='lines', name='Past measures', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Predicted'], mode='lines', name='Predicted measures', line=dict(color='red')))
    
    # Update title based on parameter
    title = 'Water Level Prediction for Sogndalsvatn' if parameter == "1000" else 'Inflow Prediction for Sogndalsvatn'
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title=data_column.capitalize()) 
    fig.update_layout(
    title=title,
    xaxis_title='Date',
    yaxis_title=data_column.capitalize(),
    yaxis=dict(range=[0.9, 2.8] if parameter == "1000" else [0, 150])  # Adjust range based on parameter
)

    
    return fig

#def plot_predictions(dataset, y_pred):
#    future_date_range = pd.date_range(end=dataset.index[-1], periods=FORECAST_HORIZON + 1, freq='3h')[1:]
#    plot_df = pd.DataFrame({'Predicted': y_pred[-1]}, index=future_date_range)
#    fig = go.Figure()

#    fig.add_trace(go.Scatter(x=dataset.index[:-FORECAST_HORIZON], y=dataset['inflow'], mode='lines', name='Past Water Level', line=dict(color='blue')))
#    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Predicted'], mode='lines', name='Predicted', line=dict(color='red', dash='dash')))
#    fig.update_layout(title='Water Level Prediction for Sogndalsvatn', xaxis_title='Date', yaxis_title='Water Level')
#    return fig

   # return fig

# Main Streamlit App
st.header("Fetching Data")
weather_data = fetch_weather_data(start_date, end_date)
inflow_data = fetch_inflow_data(station_id, parameter, start_date, end_date)


model = load_model(parameter)
if inflow_data is not None:
    st.success("Data fetched successfully!")

    st.header("Preprocessing Data")
    dataset = preprocess_data(weather_data, inflow_data, parameter)  # Pass 'parameter' here

    st.write("Data preprocessing completed!")
    
    fig_weather = go.Figure()
    fig_weather.add_trace(go.Scatter(x=dataset.index, y=dataset['tm3h1'], mode='lines', name='Temperature', line=dict(color='orange')))
    fig_weather.add_trace(go.Scatter(x=dataset.index, y=dataset['rr3h1'], mode='lines', name='Precipitation', line=dict(color='blue')))
    fig_weather.update_layout(title='Past and forecasted measures',
                           xaxis_title='Date',
                           yaxis_title='Value')
    st.plotly_chart(fig_weather)

    st.header("Making Predictions")
    X, y = prepare_sequences(dataset, parameter)
    st.success("Predictions completed!")
    y_pred = model.predict(X)
    st.success("Predictions completed!")

    st.header("Prediction Results")
    fig = plot_predictions(dataset, y_pred, parameter)
    st.plotly_chart(fig)
else:
    st.error("Failed to fetch inflow data.")



# Extract two-steps-ahead predictions and actual values
Day_ahead_predictions = y_pred[:, 4]
Actual_day_ahead = y[:, 4]

# Create a date range for the test set
date_range = pd.date_range(start=start_date, periods=len(Day_ahead_predictions), freq='3H')
shifted_date_range = date_range + timedelta(hours=12)

# Create a DataFrame for plotting
plot_df = pd.DataFrame({
    'Actual': Actual_day_ahead,
    'Predicted': Day_ahead_predictions
}, index=shifted_date_range)


# Create the plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=plot_df.index,
    y=plot_df['Actual'],
    mode='lines',
    name='Actual 12 hours ahead - historic data',
    line=dict(color='blue')
))

fig.add_trace(go.Scatter(
    x=plot_df.index,
    y=plot_df['Predicted'],
    mode='lines',
    name='Predicted 12 hours ahead - historic data',
    line=dict(color='red')  # You can customize the line style
))

# Update x-axis to display dates correctly
fig.update_xaxes(
    tickmode='auto',  # Automatically determine tick positions
    nticks=10,        # Set the approximate number of ticks you want
    tickformat="%Y-%m-%d"  # Format the tick labels as 'YYYY-MM-DD'
)

# Render in Streamlit app
st.plotly_chart(fig)

from sklearn.metrics import r2_score

# Calculate R² score
r2 = r2_score(Actual_day_ahead, Day_ahead_predictions)

# Display R² score in the Streamlit app
st.header("Model Performance")
from sklearn.metrics import mean_absolute_error

# Calculate MAE
mae = mean_absolute_error(Actual_day_ahead, Day_ahead_predictions)

# Display MAE in the Streamlit app
#st.write(f"Mean Absolute Error (MAE): {mae:.2f}") 
if parameter == "1000": st.write(f"The average prediction error in cm is: {mae*100:.2f}")
else: st.write(f"The average prediction error in m3/s is: {mae:.2f}")
st.write(f"R² score (Day_ahead_predictions): {r2:.2f}")
