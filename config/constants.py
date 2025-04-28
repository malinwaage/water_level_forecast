import os

# Constants
API_KEY = os.environ.get('NVE_API_KEY', 'your_api_key_here')
BASE_URL = 'https://hydapi.nve.no/api/v1/Observations'
SEQUENCE_LENGTH = 24
FORECAST_HORIZON = 24
