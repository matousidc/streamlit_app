import json

import streamlit as st
import streamlit.components.v1 as components

# setup page style
st.set_page_config(
    page_title="My app-Cycling statistics",
    page_icon="🧊",
    initial_sidebar_state="auto",
    menu_items={
        # 'Get Help': 'https://www.extremelycoolapp.com/help',
        # 'Report a bug': "https://www.extremelycoolapp.com/bug",
        "About": "Made by https://github.com/matousidc!"
    },
)


def load_charts(source: str) -> str:
    """Func to load charts"""
    with open(source) as f:
        return f.read()


html_roads = load_charts("./charts/deck_roads.html")
html_frequent = load_charts("./charts/deck_frequent.html")
html_clusters = load_charts("./charts/deck_clusters.html")
progress_chart = load_charts("./charts/progress_chart.json")
regression_chart = load_charts("./charts/regression.json")
cbar_clusters = load_charts("./charts/cbar_clusters.json")
cbar_roads = load_charts("./charts/cbar_roads.json")
cbar_frequent = load_charts("./charts/cbar_frequent.json")

# ============================= writing to streamlit
st.title("Brno cycling statistics")
st.markdown(
    "The data was taken from [data.brno.cz](https://data.brno.cz/), and two datasets were used: one about cycling "
    "infrastructure and the other about the 'To work by bike' event that takes place every May.")
st.markdown("### Progress of total cycling infrastructure length with fitted linear regression")
st.vega_lite_chart(json.loads(regression_chart))

st.markdown("### Length of new cycling infrastructure built each year")
st.vega_lite_chart(json.loads(progress_chart))

st.markdown("### Cycling infrastructure built over time")
components.html(html_roads, height=500, scrolling=True)
st.vega_lite_chart(json.loads(cbar_roads), use_container_width=True)

st.markdown("### The most frequent segments during 'To work by bike' event")
components.html(html_frequent, height=500, scrolling=True)
st.vega_lite_chart(json.loads(cbar_frequent), use_container_width=True)

st.markdown("### Frequency of cyclists in areas during 'To work by bike'")
st.markdown(
    "Brno has been clustered into 12 areas, which are colored based on the total frequency of cyclists in each area. "
    "This was done using the Birch clustering algorithm.")
components.html(html_clusters, height=500, scrolling=True)
st.vega_lite_chart(json.loads(cbar_clusters), use_container_width=True)
