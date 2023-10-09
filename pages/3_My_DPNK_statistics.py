import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import altair as alt
from sqlalchemy import create_engine, text

load_dotenv(override=True)

# setup page style
st.set_page_config(
    page_title="My app-My DPNK statistics",
    page_icon="🧊",
    initial_sidebar_state="auto",
    menu_items={"About": "Made by https://github.com/matousidc!"},
)

# fetching data
with st.spinner():
    connection_string = f"mysql+mysqlconnector://{os.getenv('USERNAME')}:{os.getenv('PASSWORD')}@{os.getenv('HOST')}/" \
                        f"{os.getenv('DATABASE')}?ssl_ca=/etc/ssl/cert.pem"
    engine = create_engine(connection_string)
    with engine.connect() as conn:
        df = pd.read_sql_query(text('SELECT * FROM strava_dpnk;'), con=conn)

# preparing dfs
df_2023 = df.iloc[df[df['name'].str.strip().str.lower() == 'dpnk'].index]
df_years = df[(df['name'].str.strip().str.lower() == 'work commute') |
              (df['name'].str.strip().str.lower() == 'dpnk')].copy()
df_years['start_date'] = pd.to_datetime(df_years["start_date"])
df_years = df_years[df_years["start_date"].dt.hour < 12]  # only mornings
df_years['date'] = pd.to_datetime(df_years['start_date'].dt.strftime("2000-%m-%d"))
df_years["year"] = df_years["start_date"].map(lambda x: x.strftime("%Y"))

# Charts
chart = (alt.Chart(df_2023).mark_line(color="orange", point={"filled": True, "size": 50, "color": "orange"})
         .encode(x=alt.X('start_date:T', title="Date"),
                 y=alt.Y('moving_time:Q', scale=alt.Scale(domain=[20, 26]), title="Moving time [minutes]"),
                 tooltip=['moving_time', alt.Tooltip('start_date:T')]))
reg = chart.transform_regression('start_date', 'moving_time').mark_line(color='red')
layer_chart = alt.layer(chart, reg).properties(width=800, height=400)

chart_years = (alt.Chart(df_years).mark_line(point=True)
               .encode(x=alt.X('date:T', title="Date"),
                       y=alt.Y('moving_time:Q', scale=alt.Scale(domain=[20, 27]), title="Moving time [minutes]"),
                       color=alt.Color("year", scale=alt.Scale(scheme='magma'), title="Year"),
                       tooltip=['moving_time', alt.Tooltip('start_date:T')])).properties(width=800,
                                                                                         height=400)

# defining frontend
st.title("My DPNK statistics")
st.markdown("### DPNK 2023")
st.altair_chart(layer_chart)
st.markdown("### DPNK all years")
st.altair_chart(chart_years)
