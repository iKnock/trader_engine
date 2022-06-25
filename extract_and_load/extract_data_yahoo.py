import yfinance as yf
import requests
from bs4 import BeautifulSoup
import pandas as pd


def read_candle_yahoo(ticker, yperiod, interval):
    ohlcv_data = {}
    temp = yf.download(ticker, period=yperiod, interval=interval)
    # temp.dropna(how="any", inplace=True)
    temp.dropna(axis=0, how='any', inplace=True)  # inplace true makes it substract from the data frame
    ohlcv_data[ticker] = temp
    return ohlcv_data


# ===============================Using crawler============================
def read_crypto_summary_page(url, headers):
    page = requests.get(url, headers=headers)
    page_content = page.content
    soup = BeautifulSoup(page_content, "html.parser")
    return soup


def crypto_summary(soup):
    tbl = soup.find_all("table", {"class": "W(100%)"})
    table_vals = {}
    for i in tbl:
        rows = i.find_all('tr')
        index = 0
        for row in rows:
            print(row.get_text(separator='|').split("|"))
            table_vals[index] = row.get_text(separator='|').split("|")
            index += 1
    return table_vals


def format_crypto_summary_df(summary_table):
    crypto_summary_df = pd.DataFrame(summary_table.values())
    # remove the last two col which are 52 week range and day chart which are none
    crypto_summary_df.drop(crypto_summary_df.columns[[-1, -2]], axis=1, inplace=True)

    new_header = crypto_summary_df.iloc[0]  # grab the first row for the header
    crypto_summary_df = crypto_summary_df[1:]  # take the data less the header row
    crypto_summary_df.columns = new_header  # set the header row as the df header
    return crypto_summary_df


def get_yahoo_hourl_candle():
    temp = yf.download("GBPJPY=X", period="60d", interval="15m")
    tickers = ["GBP/JPY"]
    yfinance_ohlcv = {}
    for ticker in tickers:
        yfinance_ohlcv[ticker] = read_candle_yahoo(ticker, '2y', '15m')
    return yfinance_ohlcv


if __name__ == '__main__':
    get_yahoo_hourl_candle()
