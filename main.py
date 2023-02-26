import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import json


def get_price(coin: str, vs_coin: str) -> str:
    """
    Simple price of coin versus another coin
    """
    params = {
        "ids": coin,
        "vs_currencies": vs_coin,
        "include_market_cap": "true",
        "include_24hr_change": "true",
    }
    r = requests.get("https://api.coingecko.com/api/v3/simple/price", params=params)
    # coins and vs currencies lists
    print(r.json())
    # print(f"price: {r.json()[coin][vs_coin]}")
    return r.json()[coin][vs_coin]


def coin_names():  # TODO: save coins into pickle, api time limit, check like once a day
    resp_coins = requests.get("https://api.coingecko.com/api/v3/coins/list")
    resp_vscoins = requests.get(
        "https://api.coingecko.com/api/v3/simple/supported_vs_currencies"
    )
    # print(len(resp.json()))
    # print(len(r.json()))
    return resp_coins.json(), resp_vscoins.json()


@st.cache_data
def coin_history(
    coin: str, vs_coin: str, days_back: int, interval: int | None = None
) -> pd.DataFrame:
    """
    Creates dataframe with coin history, interval is either automatic or can be specified
    """
    params = {"vs_currency": vs_coin, "days": days_back, "interval": interval}
    r = requests.get(
        f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart", params=params
    )
    df = pd.DataFrame(r.json())
    df["timestamp"] = df["prices"].apply(lambda x: x[0])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["prices"] = df["prices"].apply(lambda x: x[1])
    df["market_caps"] = df["market_caps"].apply(lambda x: x[1])
    df["total_volumes"] = df["total_volumes"].apply(lambda x: x[1])
    # print(df.head())
    return df


def plot_history(df: pd.DataFrame):
    """
    Line plot from dataframe with coin history
    """
    sns.lineplot(df, x="timestamp", y="prices")
    plt.show()


def streamlit_header():
    st.set_page_config(
        page_title="My app",
        page_icon="🧊",
        initial_sidebar_state="auto",
        menu_items={
            # 'Get Help': 'https://www.extremelycoolapp.com/help',
            # 'Report a bug': "https://www.extremelycoolapp.com/bug",
            "About": "Made by https://github.com/matousidc!"
        },
    )
    st.title("Coin chart")
    st.markdown("## Main page")
    # sidebar
    with st.sidebar:
        add_radio = st.radio(
            "Choose a shipping method", ("Standard (5-15 days)", "Express (2-5 days)")
        )
        print(add_radio)


def streamlit_plotting(df, coin, vs_coin):
    st.markdown(f"### Coin history chart: {coin}/{vs_coin}")
    st.line_chart(data=df, x="timestamp", y="prices")
    # same plot but seaborn through plt.figure
    fig = plt.figure(figsize=(10, 4))
    sns.lineplot(df, x="timestamp", y="prices")
    st.pyplot(fig)


def streamlit_options(coins: list[dict], vs_coins: list) -> tuple[str, str, int]:
    # form batching choices together, app is rerun after clicking on submit
    with st.form("my_form"):
        option = st.selectbox(
            "What coin history do you want to see?",
            tuple(f"{x['name']}-${x['symbol']}" for x in coins),
        )
        vs_option = st.selectbox("Against which coin?", tuple(vs_coins))
        interval = st.radio(
            "How far back do you want to display the history?",
            ("1 day", "1 week", "1 month", "3 months"),
        )
        interval_match = {
            x: y
            for x, y in zip(("1 day", "1 week", "1 month", "3 months"), (1, 7, 30, 90))
        }
        days_back = interval_match[interval]
        coins_dict = {f"{x['name']}-${x['symbol']}": x["id"] for x in coins}
        option = coins_dict[option]
        st.form_submit_button("Submit")
    return option, vs_option, days_back


def main():
    # price = get_price("solana", "usd")
    # plot_history(df_history)

    # pushing to streamlit
    streamlit_header()
    coins, vs_coins = coin_names()
    option, vs_option, days_back = streamlit_options(coins, vs_coins)
    df_history = coin_history(option, vs_option, days_back)
    streamlit_plotting(df_history, option, vs_option)
    print("rerun")


if __name__ == "__main__":
    main()

# df.query('@df > s and sdf <= sd'), chaining methods
