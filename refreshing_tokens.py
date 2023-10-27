#!/home/matou/projects/streamlit_app/.venv/bin/python

import os
from dotenv import load_dotenv
import requests
import pandas as pd
from sqlalchemy import create_engine

load_dotenv(override=True)

# fetching new access token (refreshing)
payload = {"client_id": os.getenv("client_id"), "client_secret": os.getenv("client_secret"),
           "grant_type": "refresh_token",
           'refresh_token': os.getenv("refresh_token")}
r = requests.post("https://www.strava.com/api/v3/oauth/token", params=payload)
access_token = r.json()['access_token']
headers = {"Authorization": f"Bearer {access_token}"}

# getting activities
activities_url = "https://www.strava.com/api/v3/athlete/activities"
payload = {"per_page": 200}
r2 = requests.get(activities_url, params=payload, headers=headers)

# creating df
activities = []
for x in r2.json():
    activities.append(
        (
            x['name'], x['start_date_local'], x['distance'] / 1000, round(x['moving_time'] / 60, 2),
            round(x['elapsed_time'] / 60, 2),
            x['total_elevation_gain'], x['average_speed'] * 3.6, x['max_speed'] * 3.6))
df = pd.DataFrame(activities,
                  columns=['name', 'start_date', 'distance', 'moving_time', 'elapsed_time', 'elev_gain', 'avg_speed',
                           'max_speed'])

# saving df
df.to_pickle("/home/matou/projects/streamlit_app/activities_df.pkl")
connection_string = f"mysql+mysqlconnector://{os.getenv('USERNAME')}:{os.getenv('PASSWORD')}@{os.getenv('HOST')}/" \
                    f"{os.getenv('DATABASE')}?ssl_ca=/etc/ssl/cert.pem"
engine = create_engine(connection_string)
with engine.begin() as conn:
    num_rows = df.to_sql('strava_dpnk', con=conn, if_exists='append', index=False)
    print('num of rows affected:', num_rows)
