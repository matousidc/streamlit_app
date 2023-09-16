from dotenv import load_dotenv
import os
import streamlit as st
from sqlalchemy import create_engine, text, types  # use for pandas queries
import pandas as pd
import altair as alt


def sqlalchemy_queries(choice: list) -> pd.DataFrame:
    connection_string = f"mysql+mysqlconnector://{os.getenv('USERNAME')}:{os.getenv('PASSWORD')}@{os.getenv('HOST')}/" \
                        f"{os.getenv('DATABASE')}?ssl_ca=/etc/ssl/cert.pem"
    engine = create_engine(connection_string)
    query = f"SELECT * FROM jobs WHERE "
    for i, skill in enumerate(choice):
        if i == 0:
            query += f"skills LIKE '%{skill}%'"
        else:
            query += f" AND skills LIKE '%{skill}%'"
    with engine.connect() as conn:  # for select query
        # df_db = pd.read_sql_query(text(f"SELECT * FROM jobs WHERE skills LIKE '%{choice[0]}%' AND "
        #                                f"skills LIKE '%{choice[1]}%';"), con=conn)
        df_db = pd.read_sql_query(text(query), con=conn)
        print(df_db)
    return df_db


def db_fetch_all() -> pd.DataFrame:
    """Fetch entire jobs table from DB"""
    connection_string = f"mysql+mysqlconnector://{os.getenv('USERNAME')}:{os.getenv('PASSWORD')}@{os.getenv('HOST')}/" \
                        f"{os.getenv('DATABASE')}?ssl_ca=/etc/ssl/cert.pem"
    engine = create_engine(connection_string)
    query = f"SELECT * FROM jobs"
    with engine.connect() as conn:  # for select query
        df_db = pd.read_sql_query(text(query), con=conn)
    return df_db


load_dotenv(override=True)
possible_skills = ['python', 'git', 'pandas', 'pyspark', 'scikit-learn', 'jenkins', 'numpy', 'bash', 'linux', 'sql',
                   'sklearn', 'pytorch', 'tensorflow', 'keras', 'jira']

with st.form("my_form"):
    option = st.multiselect('With what skill would you like to see jobs?', possible_skills)  # returns list
    st.write('You selected:', option)
    st.form_submit_button("Submit")
    if option:
        df = sqlalchemy_queries(option)
        st.dataframe(df, width=5000)

# ************************** plots
#
df_all = db_fetch_all()
# if else needed, because error on 0 count for a skill
counts = [xx[True] if (xx := df_all['skills'].apply(lambda x: skill in x).value_counts()).size > 1 else 0 for skill in
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
st.markdown("### What skills are the most wanted?")
st.altair_chart(chart, use_container_width=True)
