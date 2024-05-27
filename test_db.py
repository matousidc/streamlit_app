from dotenv import load_dotenv
import os
import datetime
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as exp_con
from selenium.webdriver.chrome.service import Service
from sqlalchemy import create_engine, text, types  # use for pandas queries

load_dotenv(override=True)
connection_string = f"mysql+mysqlconnector://{os.getenv('USERNAME')}:{os.getenv('PASSWORD')}@{os.getenv('HOST')}/" \
                    f"{os.getenv('DATABASE')}?ssl_ca=/etc/ssl/cert.pem"
engine = create_engine(connection_string)

with engine.connect() as conn:  # for select query
    df_db = pd.read_sql_query(text(f'SELECT * FROM jobs;'), con=conn)
    print('fetching df_db shape:', df_db.shape)

print(df_db)
df_db.to_pickle("df_scraped_jobs.pkl")
