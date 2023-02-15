import requests
import pandas
import streamlit


def get_price(coin, vs_coin):
    params = {"ids": coin, "vs_currencies": vs_coin}
    r = requests.get("https://api.coingecko.com/api/v3/simple/price", params=params)
    # coins and vs currencies lists
    # r = requests.get("https://api.coingecko.com/api/v3/coins/list")
    # r = requests.get("https://api.coingecko.com/api/v3/simple/supported_vs_currencies")
    print(r.json())
    print(f"price: {r.json()[coin][vs_coin]}")
    return r.json()[coin][vs_coin]


def main():
    get_price("solana", "usd")


if __name__ == '__main__':
    main()


