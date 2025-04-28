import plotly.graph_objects as go

def plot_predictions(dataset, y_pred, parameter):
    future_date_range = pd.date_range(end=dataset.index[-1], periods=FORECAST_HORIZON + 1, freq='3h')[1:]
    plot_df = pd.DataFrame({'Predicted': y_pred[-1]}, index=future_date_range)
    fig = go.Figure()

    data_column = 'waterlevel' if parameter == "1000" else 'discharge'
    fig.add_trace(go.Scatter(x=dataset.index[:-FORECAST_HORIZON], y=dataset[data_column], mode='lines', name='Past measures', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Predicted'], mode='lines', name='Predicted measures', line=dict(color='red')))
    
    title = 'Water Level Prediction for Sogndalsvatn' if parameter == "1000" else 'Inflow Prediction for Sogndalsvatn'
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title=data_column.capitalize())
    
    return fig

def plot_weather_data(dataset, weather_type):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataset.index, y=dataset[f'tm3h{weather_type}'], mode='lines', name='Temperature', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=dataset.index, y=dataset[f'rr3h{weather_type}'], mode='lines', name='Precipitation', line=dict(color='blue')))
    fig.update_layout(title='Temperature and Precipitation',
                      xaxis_title='Date',
                      yaxis_title='Value')
    return fig
