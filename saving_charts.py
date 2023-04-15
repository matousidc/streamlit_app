import json

import altair as alt
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk
import statsmodels.formula.api as smf
from matplotlib import cm
from sklearn.cluster import Birch


def load_df_dpnk() -> pd.DataFrame:
    """Loads 'to work by bike' dataset"""
    df = pd.read_csv(r"data_project3\dpnk.csv")
    return df


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


def infrastructure_progress(roads: pd.DataFrame) -> pd.DataFrame:
    """Prepares dataframe with length of built infrastucture"""
    df = roads[["rok_realizace", "delka"]].groupby(["rok_realizace"]).sum().reset_index()
    df.loc[-1] = [1996.0, 0.0]  # adding missing value for that year
    df.sort_values(by=["rok_realizace"], inplace=True, ignore_index=True)
    df['delka'] = df['delka'].round()
    df["total_length"] = df["delka"].cumsum() / 1000
    df.columns = ["year", "length", "total_length"]
    return df


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


def create_colorbar(df_in: pd.DataFrame, by: str, col_name: str, num: int) -> alt.Chart:
    """Custom colorbar using altair Chart, used for maps"""
    array = np.linspace(df_in[by].min(), df_in[by].max(), num).round()
    df = pd.DataFrame({col_name: array})
    colorbar = alt.Chart(df).mark_rect().encode(
        x=alt.X(f'{col_name}:O',
                axis=alt.Axis(values=[df[col_name].min(), df[col_name].max()], labelAngle=0, title=None)),
        color=alt.Color(col_name, scale=alt.Scale(scheme="plasma"), legend=None)
    ).properties(height=70)
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
cbar_roads = create_colorbar(geo_roads, by='year', col_name='year', num=10)
cbar_frequent = create_colorbar(geo_frequent, by='mean_years', col_name='frequency', num=10)
cbar_clusters = create_colorbar(geo_clusters, by='birch_cmap', col_name='frequency', num=12)

chart = (
    alt.Chart(df_realizace)
    .mark_point(color="red", size=60, filled=True)
    .encode(
        x=alt.X("year:O", title="Year"),
        y=alt.Y("total_length:Q", scale=alt.Scale(domain=(-20, 140))),
    )
)

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
regression = alt.layer(chart, reg_chart)
# ==================================== maps

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

# ============================= saving charts
deck_roads.to_html('./charts/deck_roads.html')
deck_frequent.to_html('./charts/deck_frequent.html')
deck_clusters.to_html('./charts/deck_clusters.html')

progress_chart.save('./charts/progress_chart.json')
regression.save('./charts/regression.json')
# chart.save('./charts/chart.json')
# reg_chart.save('./charts/reg_chart.json')

cbar_roads.save('./charts/cbar_roads.json')
cbar_frequent.save('./charts/cbar_frequent.json')
cbar_clusters.save('./charts/cbar_clusters.json')
