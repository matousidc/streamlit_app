from dotenv import load_dotenv
import os
import mysql.connector as sql  # use for normal queries
from sqlalchemy import create_engine, text, types  # use for pandas queries
import pandas as pd

load_dotenv(override=True)

# connection = sql.connect(
#     host=os.getenv("HOST"),
#     user=os.getenv("USERNAME"),
#     passwd=os.getenv("PASSWORD"),
#     db=os.getenv("DATABASE"),
#     ssl_ca="/etc/ssl/cert.pem"
#     # autocommit=True,
#     # ssl_mode="VERIFY_IDENTITY",
# )
connection_string = f"mysql+mysqlconnector://{os.getenv('USERNAME')}:{os.getenv('PASSWORD')}@{os.getenv('HOST')}/" \
                    f"{os.getenv('DATABASE')}?ssl_ca=/etc/ssl/cert.pem"
engine = create_engine(connection_string)

data = {'Name': ['Tom', 'nick', 'krish', 'jack'],
        'Age': [20, 21, 19, 18]}
df = pd.DataFrame(data)
print(df)
# dff = pd.DataFrame(engine.connect().execute(text('SELECT * FROM categories;')))
# with engine.connect() as conn: # for select query
#     # df2 = pd.read_sql_query(text('SELECT * FROM products;'), con=conn)

#     print('bruh:', qq)

with engine.begin() as conn:  # for inserting
    qq = df.to_sql('test', con=conn, if_exists='append', index=False)
    # dtype={'Name': types.VARCHAR(255), 'Age': types.INT}
    print('bruh:', qq)

# TODO: insert rows into test table using console then try overwrite


# cursor = connection.cursor()
# cursor.execute('SELECT * FROM categories;')
# m = cursor.fetchall()
# print(m)
# cursor.execute("SELECT * FROM products;")
# ww = cursor.fetchall()
# print(ww)


# print(df)
# df.to_sql('test', con=connection)
# xx = pd.read_sql("SELECT * FROM categories", connection)
# print(xx)
