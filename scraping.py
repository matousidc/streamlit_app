import selenium.common.exceptions
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
from sqlalchemy import create_engine, text, types  # use for pandas queries


def number_of_pages(driver: webdriver) -> int:
    """Finds number of pages from number buttons in footer of the page"""
    driver.get(f"https://jobs.techloop.io/?locations=ChIJEVE_wDqUEkcRsLEUZg-vAAQ&query=python&page=0")
    wait = WebDriverWait(driver, 10)
    wait.until(exp_con.presence_of_element_located((By.CSS_SELECTOR,
                                                    "button.MuiButtonBase-root.MuiPaginationItem-root"
                                                    ".MuiPaginationItem-sizeMedium.MuiPaginationItem-text"
                                                    ".MuiPaginationItem-circular.MuiPaginationItem-textPrimary"
                                                    ".MuiPaginationItem-page.css-d1721y")))
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    elements = soup.find_all('button',
                             class_='MuiButtonBase-root MuiPaginationItem-root MuiPaginationItem-sizeMedium '
                                    'MuiPaginationItem-text MuiPaginationItem-circular MuiPaginationItem-textPrimary '
                                    'MuiPaginationItem-page css-d1721y')
    return int(elements[-1].text)


def extracting_elements(driver: webdriver, num_pages: int) -> list:
    """Extracts page source in html, find wanted elements"""
    # needed to use selenium, so the page loads with JS and gives full page source
    elements_list = []
    for num in range(num_pages):
        # brno + python filters
        if num:
            driver.get(f"https://jobs.techloop.io/?locations=ChIJEVE_wDqUEkcRsLEUZg-vAAQ&query=python&page={num}")
        # wait until page loads fully
        wait = WebDriverWait(driver, 10)
        wait.until(exp_con.presence_of_element_located((By.CSS_SELECTOR,
                                                        'a.MuiTypography-root.MuiTypography-inherit.MuiLink-root'
                                                        '.MuiLink-underlineAlways.css-1mnxei8')))
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        elements = soup.find_all('a',
                                 class_='MuiTypography-root MuiTypography-inherit MuiLink-root MuiLink-underlineAlways '
                                        'css-1mnxei8')
        if not num:
            elements_list = elements
        else:
            for x in elements:
                elements_list.append(x)
    return elements_list


def making_df(elements: list) -> pd.DataFrame:
    """Creates dataframe from scraped html elements"""
    data = {'title': [x.text for x in elements], 'link': [x['href'] for x in elements],
            'date_of_scraping': datetime.date.today(), 'skills': None, 'pay': None}
    return pd.DataFrame.from_dict(data)


def merging_dfs(df: pd.DataFrame, df_db: pd.DataFrame) -> pd.DataFrame:
    """Does union merge on new(df) and fetched from database dataframe(df_db)"""
    if not df_db.empty:  # TODO: not ideal, have some edge cases
        df = pd.merge(df_db, df, how='outer')  # should prefer df_db rows
        df.sort_values(by=['date_of_scraping'], ascending=False, ignore_index=True, inplace=True)
        df.drop_duplicates('link', keep='last', inplace=True)  # only unique rows by 'link' column, keeping older date
        df = df.reset_index(drop=True)

    return df


def individual_jobs(driver: webdriver, df: pd.DataFrame) -> pd.DataFrame:
    """Searching individual jobs, extracting info into dataframe"""
    for row in df.iterrows():  # row == tuple(index, pd.Series of that row)
        site_loaded = False  # flag
        if pd.isna(row[1].skills) or row[1].skills == '':
            site_loaded = True
            print('new row:', row[0])
            driver.get(row[1].link)
            wait = WebDriverWait(driver, 10)
            try:
                wait.until(exp_con.presence_of_element_located((By.CSS_SELECTOR,
                                                                "p.MuiTypography-root.MuiTypography-body1.css-1rlo451")))
            except selenium.common.exceptions.TimeoutException:
                print('exception')
                continue
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            element = soup.find_all('p',
                                    class_='MuiTypography-root MuiTypography-body1 css-1rlo451')
            # possible more HTML parts with same class name or job listing no longer exists -> empty list
            if isinstance(element, list):
                try:
                    element = element[-1]
                except:
                    if not element:  # TODO: remove this row, job listing no longer exists
                        print('bruh', row)  # not sure what is being removed
                        print(df.shape)
                        df.drop(row[0], inplace=True)
                        print(df.shape)
                        continue
                    print("len element:", len(element))
                    print(element)
            req_skills = parsing_job_info(str(element))
            if not req_skills:  # if no skills, remove that row
                df.drop(row[0], inplace=True)
                continue
            df.loc[row[0], 'skills'] = ','.join(req_skills)
        # adding values to pay column
        if pd.isna(row[1].pay) or row[1].pay == '':
            if not site_loaded:
                driver.get(row[1].link)
                wait = WebDriverWait(driver, 10)
                wait.until(exp_con.presence_of_element_located((By.CSS_SELECTOR,
                                                                "p.MuiTypography-root.MuiTypography-body1.css-1rlo451")))
                soup = BeautifulSoup(driver.page_source, 'html.parser')
            pay_elements = soup.find_all('div',
                                         class_="css-fh68q9")
            for x in pay_elements:
                if 'Employee' in x.text:
                    df.loc[row[0], 'pay'] = x.text.split('Employee')[-1]
                    break
    df.reset_index(drop=True, inplace=True)
    return df


def parsing_job_info(element: str) -> list:
    """Parsing job requirements from string of a html"""
    possible_skills = ['python', 'git', 'pandas', 'pyspark', 'scikit-learn', 'jenkins', 'numpy', 'bash', 'linux', 'sql',
                       'sklearn', 'pytorch', 'tensorflow', 'keras', 'jira']
    req_skills = [skill for skill in possible_skills if skill in str.lower(element)]
    return req_skills


def db_connection(table: str, df: pd.DataFrame = None, insert: bool = None,
                  select: bool = None) -> pd.DataFrame | bool | None:
    """Makes connection to database, writes df to table"""
    load_dotenv(override=True)
    connection_string = f"mysql+mysqlconnector://{os.getenv('USERNAME')}:{os.getenv('PASSWORD')}@{os.getenv('HOST')}/" \
                        f"{os.getenv('DATABASE')}?ssl_ca=/etc/ssl/cert.pem"
    engine = create_engine(connection_string)
    if insert:
        with engine.begin() as conn:  # inserting to database
            rows_num = df.to_sql(table, con=conn, if_exists='replace', index=False)
            print('db: num of rows affected:', rows_num)
        return True
    if select:
        with engine.connect() as conn:  # for select query
            df_db = pd.read_sql_query(text(f'SELECT * FROM {table};'), con=conn)
            print('fetching df_db shape:', df_db.shape)
        return df_db
    return None


def outlier():
    """Helper function for testing outlier"""
    driver = webdriver.Firefox()
    driver.get('https://jobs.techloop.io/job/23140')
    wait = WebDriverWait(driver, 10)
    wait.until(exp_con.presence_of_element_located((By.CSS_SELECTOR,
                                                    "p.MuiTypography-root.MuiTypography-body1.css-1rlo451")))
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    # element = soup.find_all('div',
    #                         class_="css-fh68q9") # pay part
    # for x in element:
    #     if 'Employee' in x.text:
    #         pay = x.text.split('Employee')[-1]
    element = soup.find_all('p',
                            class_='MuiTypography-root MuiTypography-body1 css-1rlo451')
    req_skills = parsing_job_info(str(element))
    print(req_skills)


def pandas_show_options(rows=None, columns=None, width=None):
    """Sets params for printing df"""
    if rows:
        pd.set_option('display.max_rows', rows)
    if columns:
        pd.set_option('display.max_columns', columns)
    if width:
        pd.set_option('display.width', width)


def main():
    pandas_show_options(columns=5, width=1000)
    driver = webdriver.Firefox()
    num_pages = number_of_pages(driver)
    elements_list = extracting_elements(driver, num_pages)
    df = making_df(elements_list)
    df_db = db_connection(table='jobs', select=True)
    df = merging_dfs(df, df_db)
    df = individual_jobs(driver, df)
    print(df)
    driver.quit()
    db_connection(table='jobs', df=df, insert=True)
    df.to_pickle(Path(Path.cwd(), 'jobs_df.pkl'))


def main2():
    pandas_show_options(columns=5, width=1000)
    driver = webdriver.Firefox()
    num_pages = number_of_pages(driver)
    elements_list = extracting_elements(driver, num_pages)
    df = making_df(elements_list)
    df_db = db_connection(table='jobs2', select=True)
    df = merging_dfs(df, df_db)
    df = individual_jobs(driver, df)
    print(df)
    driver.quit()
    db_connection(table='jobs2', df=df, insert=True)
    df.to_pickle(Path(Path.cwd(), 'jobs2_df.pkl'))


if __name__ == "__main__":
    main()
    # main2()
    # outlier()
