import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import statsmodels.formula.api as smf
import geopandas
import pydeck as pdk


# setup page style
st.set_page_config(
    page_title="My app-cycling_data",
    page_icon="🧊",
    initial_sidebar_state="auto",
    menu_items={
        # 'Get Help': 'https://www.extremelycoolapp.com/help',
        # 'Report a bug': "https://www.extremelycoolapp.com/bug",
        "About": "Made by https://github.com/matousidc!"
    },
)
st.title("Brno cycling statistics")


@st.cache_data
def load_df_dpnk():
    """Loads 'to work by bike' dataset"""
    df_dpnk = pd.read_csv("data_project3\dpnk.csv")
    return df_dpnk


@st.cache_data
def load_df_roads():
    """Loads and cleans cycling infrastructure dataset"""
    df_roads = pd.read_csv("data_project3/Cykloopatreni.csv")
    # length in metres
    # deletes rows with year == 0
    df_roads = df_roads.drop(df_roads[df_roads["rok_realizace"] == 0].index)
    # year into int type
    df_roads["rok_realizace"] = df_roads["rok_realizace"].astype("Int64")
    # zrušená infrastruktura vyjádřena zápornou délkou, příprava pro lineární regresi
    df_roads.loc[df_roads[df_roads["typ_opatreni"] == "Úsek byl zrušen"].index, "delka"] = (
        df_roads["delka"].loc[df_roads[df_roads["typ_opatreni"] == "Úsek byl zrušen"].index] * -1
    )
    df_roads.loc[df_roads[df_roads["typ_opatreni"] == "Vjezd cyklistům zakázán"].index, "delka"] = (
        df_roads["delka"].loc[df_roads[df_roads["typ_opatreni"] == "Vjezd cyklistům zakázán"].index] * -1
    )
    return df_roads


@st.cache_data
def infrastructure_progress():
    df_realizace = df_roads[["rok_realizace", "delka"]].groupby(["rok_realizace"]).sum().reset_index()
    df_realizace.loc[-1] = [1996.0, 0.0]
    df_realizace.sort_values(by=["rok_realizace"], inplace=True, ignore_index=True)
    df_realizace["total_length"] = df_realizace["delka"].cumsum() / 1000
    df_realizace.columns = ["year", "length", "total_length"]
    return df_realizace


@st.cache_data
def get_reg_fit(data: pd.DataFrame, yvar: str, xvar: str, alpha=0.05) -> tuple[pd.DataFrame, alt.Chart]:
    # Grid for predicted values
    x = data.loc[pd.notnull(data[yvar]), xvar]
    grid = np.arange(x.min(), x.max() + 0.1)
    predictions = pd.DataFrame({xvar: grid})
    data[xvar] = data[xvar].astype("float64")

    # Fit model, get predictions
    model = smf.ols(f"{yvar} ~ {xvar}", data=data).fit()
    model_predict = model.get_prediction(predictions[xvar])
    predictions[yvar] = model_predict.summary_frame()["mean"]
    predictions[["ci_low", "ci_high"]] = model_predict.conf_int(alpha=alpha)
    # predictions[xvar] = predictions[xvar].astype("Int64")

    # Build chart
    reg = alt.Chart(predictions).mark_line(color="darkorange", opacity=0.8).encode(x=f"{xvar}:O", y=f"{yvar}:Q")
    ci = (
        alt.Chart(predictions)
        .mark_errorband(color="darkorange", opacity=0.2)
        .encode(
            x=f"{xvar}:O",
            y=alt.Y("ci_low:Q", title="Total infrastructure length [km]"),
            y2="ci_high:Q",
        )
    )
    chart = ci + reg
    return predictions, chart


# @st.cache_data
def prepare_geodata():
    # geo_dpnk = geopandas.read_file("./data_project3/dpnk_json.geojson")
    geo_dpnk = None
    geo_roads = geopandas.read_file("./data_project3/cykloopatreni_json.geojson", encoding="utf-8")
    geo_roads = geo_roads[["rok_realizace", "delka", "geometry"]]
    geo_roads = geo_roads.drop(geo_roads[geo_roads["rok_realizace"] == 0].index)
    geo_roads["color"] = geo_roads.apply(lambda x: (237, 28, 36) if x.name % 2 else (180, 0, 200, 140), axis=1)
    # geo_roads["geometry"] = geo_roads["geometry"].astype("object")
    return geo_dpnk, geo_roads  # .to_json()


# =============================================== defining plots
# scatter plot with regression line
df_dpnk = load_df_dpnk()
df_roads = load_df_roads()
df_realizace = infrastructure_progress()
geo_dpnk, geo_roads = prepare_geodata()
chart = (
    alt.Chart(df_realizace)
    .mark_point(color="red", size=60, filled=True)
    .encode(
        x=alt.X("year:O", title="Year"),
        y=alt.Y("total_length:Q", scale=alt.Scale(domain=(-20, 140))),
    )
)
# y=alt.Y("total_length:Q", axis=alt.Axis(title="Total infrastructure length [km]")),

polynomial_fit = chart.transform_regression("year", "total_length").mark_line(color="darkorange")
progress_chart = (
    alt.Chart(df_realizace)
    .mark_bar()
    .encode(
        x=alt.X("year:O", axis=alt.Axis(title="Year")),
        y=alt.Y("length:Q", axis=alt.Axis(title="Length of new infrastructure [m]")),
        color=alt.Color("length", scale=alt.Scale(scheme="goldred"), legend=alt.Legend(title="Length [m]")),
    )
)
fit, reg_chart = get_reg_fit(df_realizace, yvar="total_length", xvar="year", alpha=0.05)
# maps
map_years = (
    alt.Chart(geo_roads)
    .mark_geoshape(filled=False)
    .encode(alt.Color("rok_realizace:Q", scale=alt.Scale(scheme="goldred"), legend=alt.Legend(title="Year")))
)

layer2 = pdk.Layer(
    type="GeoJsonLayer",
    data=geo_roads,
    pickable=True,
    # get_fill_color=[255, 255, 255],
    get_line_color="color",
    # get_width=100,
    line_width_scale=20,
)
initial_position = pdk.ViewState(latitude=49.196023157428, longitude=16.60988, zoom=11, pitch=0, bearing=0)
deck = pdk.Deck(layers=[layer2], initial_view_state=initial_position, tooltip={"text": "Built in {rok_realizace}"})
# ============================= writing to streamlit
# another arguments into alt.Chart
# x="year:O",
# y="total_length:Q",
# .configure_mark(color="red")
# .interactive()
# band = alt.Chart(chart).mark_errorband(extent="ci").encode(x="year:O", y=alt.Y("total_length:Q")) # doesnt work on transform_regression
# scheme="magma" nice but low values not visible on dark background
# rebeccapurple nice purple


st.markdown("### 'To work by bike' dataset")
st.dataframe(df_dpnk)
st.markdown("### Cycling infrastructure dataset")
st.dataframe(df_roads)
st.markdown("### Progress of total cycling infrastructure length with fitted linear regression")
st.altair_chart(chart + reg_chart, use_container_width=True)
# st.altair_chart(alt.layer(chart, polynomial_fit), use_container_width=True)
st.markdown("### Length of new cycling infrastructure built each year")
st.altair_chart(progress_chart, use_container_width=True)
st.markdown("### Map of cycling infrastructure build by year")
st.dataframe(geo_roads.astype(str))
# st.altair_chart(map_years, use_container_width=True)
st.pydeck_chart(deck)
print("bruh")
