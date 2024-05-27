#!/home/matou/projects/streamlit_app/.venv/bin/python

import os
from dotenv import load_dotenv
import requests
import pandas as pd
import sqlite3

load_dotenv(override=True)


def database(df: pd.DataFrame | None):
    conn = sqlite3.connect("local_db.db")
    if df is not None:
        df.to_sql('strava_dpnk', conn, if_exists='replace', index=False)
        return None
    else:
        df_db = pd.read_sql_query("SELECT * FROM strava_dpnk;", con=conn)
        return df_db


def strava_request() -> requests.Response:
    """fetching new access token (refreshing)"""
    payload = {"client_id": os.getenv("client_id"), "client_secret": os.getenv("client_secret"),
               "grant_type": "refresh_token",
               'refresh_token': os.getenv("refresh_token")}
    r = requests.post("https://www.strava.com/api/v3/oauth/token", params=payload)
    access_token = r.json()['access_token']
    headers = {"Authorization": f"Bearer {access_token}"}
    # getting activities
    activities_url = "https://www.strava.com/api/v3/athlete/activities"
    payload = {"per_page": 200}
    resp = requests.get(activities_url, params=payload, headers=headers)
    return resp


def creating_df(resp: requests.Response) -> pd.DataFrame:
    activities = []
    for x in resp.json():
        activities.append(
            (
                x['name'], x['start_date_local'], x['distance'] / 1000, round(x['moving_time'] / 60, 2),
                round(x['elapsed_time'] / 60, 2),
                x['total_elevation_gain'], x['average_speed'] * 3.6, x['max_speed'] * 3.6))
    df = pd.DataFrame(activities,
                      columns=['name', 'start_date', 'distance', 'moving_time', 'elapsed_time', 'elev_gain',
                               'avg_speed', 'max_speed'])
    return df


def main():
    pd.set_option("display.max_columns", 7)
    pd.set_option("display.width", 1000)
    resp = strava_request()
    df = creating_df(resp)
    # df_db = database(df=None)  # sql read
    df_db = pd.read_pickle("new_df_strava.pkl")  # first time use
    df_merged = pd.merge(df_db, df, how="outer").sort_values(by="start_date", ascending=False).reset_index(drop=True)
    database(df_merged)


if __name__ == "__main__":
    main()
