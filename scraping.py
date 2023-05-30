# div.gtm-element-visibility-job-impressions-list:nth-child(2) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > h5:nth-child(1) > a:nth-child(1)
# html body.theme-light div#app div div.jss1 div.MuiGrid-root.MuiGrid-container.MuiGrid-wrap-xs-nowrap.css-jnas21 div.MuiGrid-root.MuiGrid-item.jss3.css-1wxaqej main.jss2 div.jss16.jss19.jss23.MuiBox-root.css-0 div.gtm-element-visibility-job-impressions-list div.MuiPaper-root.MuiPaper-outlined.MuiPaper-rounded.MuiCard-root.css-121z6ut div.MuiBox-root.css-i3pbo div.MuiBox-root.css-qmhpxv div.MuiBox-root.css-0 h5.MuiTypography-root.MuiTypography-h5.css-16ddinj a.MuiTypography-root.MuiTypography-inherit.MuiLink-root.MuiLink-underlineAlways.css-1mnxei8
# /html/body/div[1]/div/div/div[2]/div/main/div[3]/div[1]/div/div[1]/div/div[1]/h5/a
import bs4.element
# <a class="MuiTypography-root MuiTypography-inherit MuiLink-root MuiLink-underlineAlways css-1mnxei8" href="https://jobs.techloop.io/job/21664" target="_blank">Python Developer</a>
import datetime
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as exp_con


def number_of_pages(driver: webdriver) -> int:
    """Finds number of pages from number buttons in page"""
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
        # print(soup)
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


def merging_dfs(df: pd.DataFrame) -> pd.DataFrame:
    """Does union merge on new and previous dataframes"""
    if Path(Path.cwd(), 'jobs_df.pkl').is_file():
        df_prev = pd.read_pickle(Path(Path.cwd(), 'jobs_df.pkl'))
        df = pd.merge(df, df_prev, how='outer')
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
            print(row[0])
            driver.get(row[1].link)
            wait = WebDriverWait(driver, 10)
            wait.until(exp_con.presence_of_element_located((By.CSS_SELECTOR,
                                                            "p.MuiTypography-root.MuiTypography-body1.css-1rlo451")))
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            element = soup.find_all('p',
                                    class_='MuiTypography-root MuiTypography-body1 css-1rlo451')
            if isinstance(element, list):
                try:
                    element = element[-1]
                except:
                    print("len element:", len(element))
                    print(element)
            req_skills = parsing_job_info(str(element))
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
    return df


def parsing_job_info(element: str) -> list:
    """Parsing job requirements from string of a html"""
    possible_skills = ['python', 'git', 'pandas', 'pyspark', 'scikit-learn', 'jenkins', 'numpy', 'bash', 'linux', 'sql',
                       'sklearn', 'pytorch', 'tensorflow', 'jira']
    req_skills = [skill for skill in possible_skills if skill in str.lower(element)]
    return req_skills


def outlier():
    """Helper function for testing outlier"""
    driver = webdriver.Firefox()
    driver.get('https://jobs.techloop.io/job/24092')
    wait = WebDriverWait(driver, 10)
    wait.until(exp_con.presence_of_element_located((By.CSS_SELECTOR,
                                                    "p.MuiTypography-root.MuiTypography-body1.css-1rlo451")))
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    element = soup.find_all('div',
                            class_="css-fh68q9")
    print(element)
    print(len(element))
    for x in element:
        if 'Employee' in x.text:
            pay = x.text.split('Employee')[-1]


def main():
    # TODO: pd.read_pickle(), compare when scraping, when scraped job already in
    driver = webdriver.Firefox()
    num_pages = number_of_pages(driver)
    elements_list = extracting_elements(driver, num_pages)
    df = making_df(elements_list)
    df = merging_dfs(df)
    df = individual_jobs(driver, df)
    print(df)
    driver.quit()
    df.to_pickle(Path(Path.cwd(), 'jobs_df.pkl'))


if __name__ == "__main__":
    main()
    # outlier()
