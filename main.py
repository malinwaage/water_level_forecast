import streamlit as st
import pandas as pd
from config.constants import SEQUENCE_LENGTH, FORECAST_HORIZON
from utils.data_fetcher import fetch_weather_data, fetch_inflow_data
from utils.data_processor import preprocess_data, prepare_sequences
from utils.model_handler import load_model
from ui.layout import setup_page_config
from ui.visualizations import plot_predictions, plot_weather_data

# Setup Streamlit page configuration
setup_page_config()

# Streamlit Inputs
st.title("🌊 Water Level/Discharge Prediction for Sogndalsvatn")
st.write("""
This application predicts water levels and inflow for Sogndalsvatn using a GRU-based deep learning model. 
It uses temperature, precipitation, and historical data for forecasts several days in advance on a three-hour basis.
""")

st.sidebar.header("User Inputs")
station_id = st.sidebar.text_input("Station ID", "77.3.0")
parameter = st.sidebar.selectbox("Parameter (1001:inflow/discharge, 1000:water-level)", ["1000", "1001"], index=0)
forecast_days = st.sidebar.slider("Forecast Days", 1, 2, 3)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("today") - pd.DateOffset(days=10))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today") + pd.DateOffset(days=forecast_days))

# Load data
try:
    weather_data = fetch_weather_data(start_date.strftime('%d.%m.%Y'), end_date.strftime('%d.%m.%Y'))
    inflow_data = fetch_inflow_data(station_id, parameter, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    # Debugging output
    st.write("Weather Data Shape:", weather_data.shape)
    st.write("Inflow Data Shape:", inflow_data.shape)

    if inflow_data.empty:
        st.error("Inflow data is empty. Please check the station ID and date range.")
        st.stop()  # Stop further execution

    st.success("Data fetched successfully!")
except Exception as e:
    st.error(f"Error fetching data: {str(e)}")
    
# Preprocess data
if inflow_data is not None:
    dataset = preprocess_data(weather_data, inflow_data, parameter)
    st.write("Data preprocessing completed!")

    # Create plots for temperature and precipitation
    st.header("Temperature and Precipitation for Sogndalsvatn")
    fig_weather = plot_weather_data(dataset, '1')  # Assuming '1' is the suffix for the first weather data
    st.plotly_chart(fig_weather)

    # Prepare sequences for prediction
    st.header("Making Predictions")
    X, y = prepare_sequences(dataset, parameter)
    model = load_model(parameter)
    y_pred = model.predict(X)
    st.success("Predictions completed!")

    # Display prediction results
    st.header("Prediction Results")
    fig_predictions = plot_predictions(dataset, y_pred, parameter)
    st.plotly_chart(fig_predictions)
else:
    st.error("Failed to fetch inflow data.")
