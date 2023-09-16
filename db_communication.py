from dotenv import load_dotenv
import os
import mysql.connector as sql  # use for normal queries
from sqlalchemy import create_engine, text, types  # use for pandas queries
import pandas as pd


def sqlalchemy_queries(select=None, insert=None):
    connection_string = f"mysql+mysqlconnector://{os.getenv('USERNAME')}:{os.getenv('PASSWORD')}@{os.getenv('HOST')}/" \
                        f"{os.getenv('DATABASE')}?ssl_ca=/etc/ssl/cert.pem"
    engine = create_engine(connection_string)

    df = pd.read_pickle(r"C:\Users\matou\PycharmProjects\projects\streamlit\jobs_df.pkl")

    # dff = pd.DataFrame(engine.connect().execute(text('SELECT * FROM categories;')))
    if select:
        with engine.connect() as conn:  # for select query
            df_db = pd.read_sql_query(text('SELECT * FROM jobs;'), con=conn)
            # df_db = pd.read_sql_query(text("SELECT * FROM jobs where skills like '%pandas%';"), con=conn)
            print(df_db.iloc[:10])

    if insert:
        with engine.begin() as conn:  # for inserting
            num_rows = df.to_sql('jobs', con=conn, if_exists='append', index=False)
            # dtype={'Name': types.VARCHAR(255), 'Age': types.INT}
            print('num of rows affected:', num_rows)


def mysql_queries():
    connection = sql.connect(
        host=os.getenv("HOST"),
        user=os.getenv("USERNAME"),
        passwd=os.getenv("PASSWORD"),
        db=os.getenv("DATABASE"),
        ssl_ca="/etc/ssl/cert.pem"
        # autocommit=True,
        # ssl_mode="VERIFY_IDENTITY",
    )
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM jobs LIMIT 5;')
    m = cursor.fetchall()
    print(m)
    cursor.execute("SELECT * FROM test;")
    ww = cursor.fetchall()
    print(ww)


def pandas_show_options(rows=None, columns=None, width=None):
    """Sets params for printing df"""
    if rows:
        pd.set_option('display.max_rows', rows)
    pd.set_option('display.max_columns', columns)
    pd.set_option('display.width', width)


def main():
    load_dotenv(override=True)
    pandas_show_options(columns=5, width=1000)
    # mysql_queries()
    sqlalchemy_queries(select=True)  # change to True for what scenario is wanted


if __name__ == "__main__":
    main()
