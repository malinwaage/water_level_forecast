import pandas as pd
import numpy as np

def preprocess_data(weather_data, inflow_data, parameter):
    dataset = weather_data.join(inflow_data)
    inflow_column = 'waterlevel' if parameter == "1000" else 'discharge'
    dataset = dataset.rename(columns={'value': inflow_column})
    dataset = dataset.mask(dataset > 1000)

    for column in dataset.columns:
        if column != inflow_column:
            dataset[column] = (dataset[column] - dataset[column].min()) / (dataset[column].max() - dataset[column].min())
            dataset = dataset.interpolate(method='linear', limit_direction='both')

    return dataset

def prepare_sequences(dataset, parameter):
    X, y = [], []
    inflow_column = 'waterlevel' if parameter == "1000" else 'discharge'

    for i in range(len(dataset) - SEQUENCE_LENGTH - FORECAST_HORIZON):
        X_seq = dataset.iloc[i:(i + SEQUENCE_LENGTH)].values
        forecast_values = dataset.iloc[i + SEQUENCE_LENGTH:i + SEQUENCE_LENGTH + FORECAST_HORIZON].values
        forecast_values[:, dataset.columns.get_loc(inflow_column)] = 0
        X_seq_with_forecast = np.concatenate([X_seq, forecast_values])
        X.append(X_seq_with_forecast)
        y.append(dataset[inflow_column].iloc[i + SEQUENCE_LENGTH:i + SEQUENCE_LENGTH + FORECAST_HORIZON].values)
    return np.array(X), np.array(y)
