import json

import altair as alt
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk
import statsmodels.formula.api as smf
import streamlit as st
from matplotlib import cm
from sklearn.cluster import Birch

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


@st.cache_data
def load_df_dpnk() -> pd.DataFrame:
    """Loads 'to work by bike' dataset"""
    df = pd.read_csv(r"data_project3\dpnk.csv")
    return df


@st.cache_data
def load_df_roads() -> pd.DataFrame:
    """Loads and cleans cycling infrastructure dataset"""
    df = pd.read_csv("data_project3/Cykloopatreni.csv")
    # length in metres
    # deletes rows with year == 0
    df = df.drop(df[df["rok_realizace"] == 0].index)
    # year into int type
    df["rok_realizace"] = df["rok_realizace"].astype("Int64")
    # zrušená infrastruktura vyjádřena zápornou délkou, příprava pro lineární regresi
    df.loc[df[df["typ_opatreni"] == "Úsek byl zrušen"].index, "delka"] = (
            df["delka"].loc[df[df["typ_opatreni"] == "Úsek byl zrušen"].index] * -1
    )
    df.loc[df[df["typ_opatreni"] == "Vjezd cyklistům zakázán"].index, "delka"] = (
            df["delka"].loc[df[df["typ_opatreni"] == "Vjezd cyklistům zakázán"].index] * -1
    )
    return df


@st.cache_data
def infrastructure_progress(roads: pd.DataFrame) -> pd.DataFrame:
    """Prepares dataframe with length of built infrastucture"""
    df = roads[["rok_realizace", "delka"]].groupby(["rok_realizace"]).sum().reset_index()
    df.loc[-1] = [1996.0, 0.0]  # adding missing value for that year
    df.sort_values(by=["rok_realizace"], inplace=True, ignore_index=True)
    df['delka'] = df['delka'].round()
    df["total_length"] = df["delka"].cumsum() / 1000
    df.columns = ["year", "length", "total_length"]
    return df


@st.cache_data
def get_reg_fit(data: pd.DataFrame, yvar: str, xvar: str, alpha=0.05) -> tuple[pd.DataFrame, alt.Chart]:
    """Creates confidence interval for altair chart"""
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
    return predictions, ci + reg


@st.cache_data
def prepare_geodata() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepares geodaframes"""
    # loading dataset through json because streamlit error with single quotes
    with open("./data_project3/dpnk_json.geojson") as file:
        dpnk = geopandas.GeoDataFrame.from_features(json.load(file))
    # dpnk = geopandas.read_file("./data_project3/dpnk_json.geojson")
    with open("./data_project3/cykloopatreni_json.geojson") as file:
        roads = geopandas.GeoDataFrame.from_features(json.load(file))
    # geo_roads = geopandas.read_file("./data_project3/cykloopatreni_json.geojson", encoding="utf-8")
    roads = roads[["rok_realizace", "delka", "geometry"]]
    roads.columns = ["year", "length", "geometry"]
    roads = roads.drop(roads[roads["year"] == 0].index)
    roads["length"] = round(roads["length"])
    # geo_roads["geometry"] = geo_roads["geometry"].astype("object")
    dpnk = dpnk[dpnk.columns[3:8].append(dpnk.columns[0:1])]
    dpnk["mean_years"] = round(dpnk[dpnk.columns[:5]].mean(axis=1)).astype("int")
    return dpnk, roads  # .to_json()


@st.cache_data
def prepare_clusters(_df_in: pd.DataFrame) -> pd.DataFrame:
    df = _df_in.copy()
    df = df.set_crs(4326)
    df["centroid"] = df["geometry"].to_crs(crs=3857).centroid  # to metres
    df["centroid"] = df["centroid"].to_crs(crs=4326)  # to degrees
    # clustering
    centroids = pd.concat([df["centroid"].x, df["centroid"].y], axis=1)
    brc = Birch(threshold=0.01, n_clusters=12, branching_factor=10).fit_predict(centroids)
    df["birch"] = brc
    df["birch_cmap"] = df["birch"].map(df[["mean_years", "birch"]].groupby(["birch"]).sum().to_dict()["mean_years"])
    # making cmap
    cmap = plt.get_cmap("plasma", 12)
    rgba_array = cm.ScalarMappable(cmap=cmap).to_rgba(list(range(12)), bytes=True)
    rgba = [tuple(int(q) for q in x) for x in rgba_array]  # values to int
    values = np.sort(df["birch_cmap"].unique())
    key = {x: y for x, y in zip(values, rgba)}
    df = df[["geometry", "mean_years", "birch_cmap"]].copy()
    df["color"] = df["birch_cmap"].map(key)
    df["birch_cmap"] = df["birch_cmap"] // 1000
    return df


def color_column(df: pd.DataFrame, by: str, cmap: str, num: int) -> pd.DataFrame:
    """Creates cmap color column with num number of bins. Bins from by column"""
    cmap = plt.get_cmap(cmap, num)
    rgba_array = cm.ScalarMappable(cmap=cmap).to_rgba(list(range(num)), bytes=True)
    rgba = [tuple(int(q) for q in x) for x in rgba_array]  # values to int
    bins = np.linspace(df[by].min(), df[by].max(), num + 1)
    df["color"] = pd.cut(df[by], bins=bins, labels=rgba)
    return df


def create_colorbar(df_in: pd.DataFrame, by: str, num: int) -> alt.Chart:
    """Custom colorbar using altair Chart, used for maps"""
    array = np.linspace(df_in[by].min(), df_in[by].max(), num).round()
    df = pd.DataFrame({'cbar': array, 'number': 1})
    colorbar = alt.Chart(df).mark_rect().encode(
        x=alt.X('cbar:O', axis=alt.Axis(values=[df['cbar'].min(), df['cbar'].max()], labelAngle=0, title=None)),
        y=alt.Y('number:O', axis=None),
        color=alt.Color("cbar", scale=alt.Scale(scheme="plasma"), legend=None)
    )  # .properties(height=80)
    return colorbar


# =============================================== defining plots
# scatter plot with regression line
df_dpnk = load_df_dpnk()
df_roads = load_df_roads()
df_realizace = infrastructure_progress(df_roads)
geo_dpnk, geo_roads = prepare_geodata()
geo_roads = color_column(df=geo_roads, by="year", cmap="plasma", num=10)
geo_frequent = geo_dpnk[geo_dpnk["mean_years"] > 300].copy()  # only roads with more than mean month value of 300
geo_frequent = color_column(df=geo_frequent, by="mean_years", cmap="plasma", num=10)
geo_clusters = prepare_clusters(geo_dpnk)
cbar_roads = create_colorbar(geo_roads, by='year', num=10)
cbar_frequent = create_colorbar(geo_frequent, by='mean_years', num=10)
cbar_clusters = create_colorbar(geo_clusters, by='birch_cmap', num=12)

chart = (
    alt.Chart(df_realizace)
    .mark_point(color="red", size=60, filled=True)
    .encode(
        x=alt.X("year:O", title="Year"),
        y=alt.Y("total_length:Q", scale=alt.Scale(domain=(-20, 140))),
    )
)
# y=alt.Y("total_length:Q", axis=alt.Axis(title="Total infrastructure length [km]")),

# polynomial_fit = chart.transform_regression("year", "total_length").mark_line(color="darkorange")
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
# ==================================== maps
# doesnt work in streamlit
# map_years = (
#     alt.Chart(geo_roads)x
#     .mark_geoshape(filled=False)
#     .encode(alt.Color("rok_realizace:Q", scale=alt.Scale(scheme="goldred"), legend=alt.Legend(title="Year")))
# )

initial_position = pdk.ViewState(latitude=49.196023157428, longitude=16.60988, zoom=11, pitch=0, bearing=0)
map_roads = pdk.Layer(
    type="GeoJsonLayer",
    data=geo_roads,
    pickable=True,
    auto_highlight=True,
    get_line_color="color",
    line_width_scale=20,
)
deck_roads = pdk.Deck(
    layers=[map_roads],
    initial_view_state=initial_position,
    tooltip={"text": "Built in {year}, length: {length} m"},
)

map_frequent = pdk.Layer(
    type="GeoJsonLayer",
    data=geo_frequent,
    pickable=True,
    auto_highlight=True,
    get_line_color="color",
    line_width_scale=20,
)
deck_frequent = pdk.Deck(
    layers=[map_frequent],
    initial_view_state=initial_position,
    tooltip={"text": "Mean frequency over the years: {mean_years} cyclists"},
)
map_cluster = pdk.Layer(
    type="GeoJsonLayer",
    data=geo_clusters,
    pickable=True,
    auto_highlight=True,
    get_line_color="color",
    line_width_scale=20,
)
deck_clusters = pdk.Deck(
    layers=[map_cluster],
    initial_view_state=initial_position,
    tooltip={"text": "Number of cyclists per mont {birch_cmap} (in thousands)"},
)

# deck_roads.to_html('deck_roads.html')
# deck_frequent.to_html('deck_frequent.html')
# deck_clusters.to_html('deck_clusters.html')

# another arguments into alt.Chart
# x="year:O",
# y="total_length:Q",
# .configure_mark(color="red")
# .interactive()
# scheme="magma" nice but low values not visible on dark background
# rebeccapurple nice purple

# ============================= writing to streamlit
st.title("Brno cycling statistics")
# TODO: info about source and datasets
# st.markdown("### 'To work by bike' dataset")
# st.dataframe(df_dpnk)
# st.markdown("### Cycling infrastructure dataset")
# st.dataframe(df_roads)
st.markdown("### Progress of total cycling infrastructure length with fitted linear regression")
st.altair_chart(chart + reg_chart, use_container_width=True)
# st.altair_chart(alt.layer(chart, polynomial_fit), use_container_width=True)
st.markdown("### Length of new cycling infrastructure built each year")
st.altair_chart(progress_chart, use_container_width=True)
st.markdown("### Map of cycling infrastructure built over time")
st.pydeck_chart(deck_roads)
st.altair_chart(cbar_roads, use_container_width=True)
st.markdown("### Map of most frequent segments during 'To work by bike'")
st.pydeck_chart(deck_frequent)
st.altair_chart(cbar_frequent, use_container_width=True)
st.markdown("### Map of areas by frequency during 'To work by bike'")
st.pydeck_chart(deck_clusters)
st.altair_chart(cbar_clusters, use_container_width=True)
print("bruh")
