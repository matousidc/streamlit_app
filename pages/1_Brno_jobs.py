from dotenv import load_dotenv
from datetime import date, timedelta
import os
import streamlit as st
from sqlalchemy import create_engine, text  # use for pandas queries
import sqlite3
import pandas as pd
import altair as alt


def db_connection(query: str) -> pd.DataFrame:
    conn = sqlite3.connect("local_db.db")
    df_db = pd.read_sql_query(query, con=conn)
    return df_db


def construct_query(query_all: bool, choice: list | None = None) -> str:
    """Creates query, chooses between 2 options"""
    if not query_all:
        query = f"SELECT * FROM scraped_jobs WHERE "
        for i, skill in enumerate(choice):
            if i == 0:
                query += f"skills LIKE '%{skill}%'"
            else:
                query += f" AND skills LIKE '%{skill}%'"
    else:
        query = f"SELECT * FROM scraped_jobs"
    return query


def skills_plot(df: pd.DataFrame) -> alt.Chart:
    # if-else needed, because error on 0 count for a skill
    counts = [xx[True] if (xx := df['skills'].apply(lambda x: skill in x).value_counts()).size > 1 else 0 for skill
              in
              possible_skills]
    # creates df from 2 lists, sorted
    df_skill_count = pd.Series({x: y for x, y in zip(possible_skills, counts)}).reset_index(name='count').sort_values(
        by=['count'], ascending=False)
    chart = (
        alt.Chart(df_skill_count)
        .mark_bar(color="red", filled=True)
        .encode(
            x=alt.X("count:Q", title="Count"),
            y=alt.Y("index:O", title="Unique skills",
                    sort=alt.EncodingSortField(field='count', op='max', order='descending')),
        )
    )
    return chart


def last_jobs(df: pd.DataFrame) -> pd.DataFrame:
    """Creates dataframe with last scraped jobs"""
    df_slice = df.loc[df["date_of_scraping"] == date.today()]
    if df_slice.empty:
        idx = 0
        while df_slice.empty:
            idx -= 1
            df_slice = df.loc[df["date_of_scraping"] == (date.today() + timedelta(days=idx))]
    return df_slice


# ============ streamlit logic
load_dotenv(override=True)
st.set_page_config(layout="wide")
possible_skills = ['python', 'git', 'pandas', 'pyspark', 'scikit-learn', 'jenkins', 'numpy', 'bash', 'linux', 'sql',
                   'sklearn', 'pytorch', 'tensorflow', 'keras', 'jira']
with st.form("my_form"):
    option = st.multiselect('Choose a skill you would like to see jobs for', possible_skills)  # returns list
    st.form_submit_button("Submit")
    if option:
        df_option = db_connection(construct_query(query_all=False, choice=option))
        st.dataframe(df_option, width=5000, column_config={"link": st.column_config.LinkColumn()})

df_all = db_connection(construct_query(query_all=True))
st.markdown("### What skills are the most wanted?")
st.altair_chart(skills_plot(df_all), use_container_width=True)
