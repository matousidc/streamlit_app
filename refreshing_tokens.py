import os
from dotenv import load_dotenv
import requests
import pandas as pd

load_dotenv(override=True)

# fetching new access token (refreshing)
payload = {"client_id": os.getenv("client_id"), "client_secret": os.getenv("client_secret"), "grant_type": "refresh_token",
           'refresh_token': os.getenv("refresh_token")}
r = requests.post("https://www.strava.com/api/v3/oauth/token", params=payload)
access_token = r.json()['access_token']
headers = {"Authorization": f"Bearer {access_token}"}

# getting activities
activities_url = "https://www.strava.com/api/v3/athlete/activities"
payload = {"per_page": 150}
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
df.to_pickle("activities_df.pkl")
