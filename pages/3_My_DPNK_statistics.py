import streamlit as st
import pandas as pd
import altair as alt
import sqlite3


def db_connection() -> pd.DataFrame:
    conn = sqlite3.connect("local_db.db")
    df_db = pd.read_sql_query('SELECT * FROM strava_dpnk;', con=conn)
    return df_db


def preparing_dfs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # filter only 'dpnk' as a title, and year as 2023
    df_2023 = df.iloc[df[df['name'].str.strip().str.lower() == 'dpnk'].index].copy()
    df_2023["start_date"] = pd.to_datetime(df_2023["start_date"])
    df_2023 = df_2023.loc[df_2023["start_date"].dt.year == 2023]
    # 2024
    df_2024 = df.iloc[df[df['name'].str.strip().str.lower() == 'dpnk'].index].copy()
    df_2024["start_date"] = pd.to_datetime(df_2024["start_date"])
    df_2024 = df_2024.loc[df_2024["start_date"].dt.year == 2024]
    # filter 'dpnk' or 'work commute' as a title and only mornings
    df_years = df[(df['name'].str.strip().str.lower() == 'work commute') |
                  (df['name'].str.strip().str.lower() == 'dpnk')].copy()
    df_years['start_date'] = pd.to_datetime(df_years["start_date"])
    df_years = df_years[df_years["start_date"].dt.hour < 12]  # only mornings
    # one column with all the years pretending to be a same year
    df_years['date'] = pd.to_datetime(df_years['start_date'].dt.strftime("2000-%m-%d"))
    df_years["year"] = df_years["start_date"].map(lambda x: x.strftime("%Y"))
    return df_2024, df_2023, df_years


def defining_charts(df_2024: pd.DataFrame, df_2023: pd.DataFrame, df_years: pd.DataFrame) -> tuple[
    alt.Chart, alt.Chart, alt.Chart]:
    # 2023
    chart_23 = (alt.Chart(df_2023).mark_line(color="orange", point={"filled": True, "size": 50, "color": "orange"})
                .encode(x=alt.X('start_date:T', title="Date"),
                        y=alt.Y('moving_time:Q', scale=alt.Scale(domain=[20, 26]), title="Moving time [minutes]"),
                        tooltip=['moving_time', alt.Tooltip('start_date:T')]))
    reg_23 = chart_23.transform_regression('start_date', 'moving_time').mark_line(color='red')
    layer_chrt_23 = alt.layer(chart_23, reg_23).properties(width=800, height=400)
    # 2024
    chart_24 = (alt.Chart(df_2024).mark_line(color="orange", point={"filled": True, "size": 50, "color": "orange"})
                .encode(x=alt.X('start_date:T', title="Date"),
                        y=alt.Y('moving_time:Q', scale=alt.Scale(domain=[20, 26]), title="Moving time [minutes]"),
                        tooltip=['moving_time', alt.Tooltip('start_date:T')]))
    reg_24 = chart_24.transform_regression('start_date', 'moving_time').mark_line(color='red')
    layer_chrt_24 = alt.layer(chart_24, reg_24).properties(width=800, height=400)
    # all years
    chrt_years = (alt.Chart(df_years).mark_line(point=True)
                  .encode(x=alt.X('date:T', title="Date"),
                          y=alt.Y('moving_time:Q', scale=alt.Scale(domain=[20, 27]), title="Moving time [minutes]"),
                          color=alt.Color("year", scale=alt.Scale(scheme='magma'), title="Year"),
                          tooltip=['moving_time', alt.Tooltip('start_date:T')])).properties(width=800,
                                                                                            height=400)
    return layer_chrt_24, layer_chrt_23, chrt_years


# ====================== streamlit logic
# setup page style
st.set_page_config(
    layout="wide",
    page_title="My app-My DPNK statistics",
    page_icon="🧊",
    initial_sidebar_state="auto",
    menu_items={"About": "Made by https://github.com/matousidc!"},
)
# fetching data
with st.spinner():
    df = db_connection()
df_2024, df_2023, df_years = preparing_dfs(df)
layer_chart_24, layer_chart_23, chart_years = defining_charts(df_2024, df_2023, df_years)

# defining frontend
st.title("My DPNK statistics")
st.markdown("### DPNK 2024")
st.altair_chart(layer_chart_24)
st.markdown("### DPNK 2023")
st.altair_chart(layer_chart_23)
st.markdown("### DPNK all years")
st.altair_chart(chart_years)
