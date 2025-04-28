import pandas as pd
import requests
from requests_html import HTMLSession
from config.constants import API_KEY, BASE_URL

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
                raise Exception(f"Error fetching data for {atr} at coordinates {coord}: Status code {r.status_code}")
    return df_weather

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
            return df.rename(columns={'value': 'waterlevel' if parameter == "1000" else 'discharge'})
    raise Exception("Failed to fetch inflow data.")
