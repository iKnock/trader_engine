import numpy as np
import transformer.indicators as indicator
import copy
import pandas as pd
import transformer.renko as rnk
import extract_and_load.load_data as ld
from datetime import datetime as dt, timezone as tz, timedelta as td
import utility.constants as const


def merge_dfs(df_intra_day):
    """Merging renko df with original ohlc df"""
    df = copy.deepcopy(df_intra_day)
    print("merging for ")
    renko = rnk.renko_dfr(df)
    renko.columns = ["Date", "open", "high", "low", "close", "uptrend", "bar_num"]
    df["Date"] = df.index
    ohlc_renko = df.merge(renko.loc[:, ["Date", "bar_num"]], how="outer", on="Date")
    ohlc_renko["bar_num"].fillna(method='ffill', inplace=True)
    ohlc_renko["macd"] = indicator.macd(ohlc_renko, 12, 26, 9)['macd']
    ohlc_renko["macd_sig"] = indicator.macd(ohlc_renko, 12, 26, 9)['signal']
    ohlc_renko["macd_slope"] = rnk.slope(ohlc_renko["macd"], 5)
    ohlc_renko["macd_sig_slope"] = rnk.slope(ohlc_renko["macd_sig"], 5)
    return ohlc_renko


def identify_signal_return_renko_macd(data_fr, ohlc_renko):
    ohlc_intraday = copy.deepcopy(data_fr)

    tickers_signal = ""
    tickers_ret = []
    df_signal = []

    for i in range(len(ohlc_intraday)):
        if tickers_signal == "":
            tickers_ret.append(0)
            if i > 0:
                if ohlc_renko["bar_num"][i] >= 2 and ohlc_renko["macd"][i] > \
                        ohlc_renko["macd_sig"][i] and ohlc_renko["macd_slope"][i] > \
                        ohlc_renko["macd_sig_slope"][i]:
                    tickers_signal = "Buy"
                elif ohlc_renko["bar_num"][i] <= -2 and ohlc_renko["macd"][i] < \
                        ohlc_renko["macd_sig"][i] and ohlc_renko["macd_slope"][i] < \
                        ohlc_renko["macd_sig_slope"][i]:
                    tickers_signal = "Sell"

            if tickers_signal == 'Sell' or tickers_signal == 'Buy':
                df_signal.append(tickers_signal)
            else:
                df_signal.append("N/A")

        elif tickers_signal == "Buy":
            tickers_ret.append(
                (ohlc_renko["CLOSE"][i] / ohlc_renko["CLOSE"][i - 1]) - 1)
            if i > 0:
                if ohlc_renko["bar_num"][i] <= -2 and ohlc_renko["macd"][i] < \
                        ohlc_renko["macd_sig"][i] and ohlc_renko["macd_slope"][i] < \
                        ohlc_renko["macd_sig_slope"][i]:
                    tickers_signal = "Sell"#Close_sell
                elif ohlc_renko["macd"][i] < ohlc_renko["macd_sig"][i] and \
                        ohlc_renko["macd_slope"][i] < ohlc_renko["macd_sig_slope"][i]:
                    tickers_signal = ""

            if tickers_signal == 'Sell' or tickers_signal == 'Buy':
                df_signal.append(tickers_signal)
            else:
                df_signal.append("Exit")

        elif tickers_signal == "Sell":
            tickers_ret.append(
                (ohlc_renko["CLOSE"][i - 1] / ohlc_renko["CLOSE"][i]) - 1)
            if i > 0:
                if ohlc_renko["bar_num"][i] >= 2 and ohlc_renko["macd"][i] > \
                        ohlc_renko["macd_sig"][i] and ohlc_renko["macd_slope"][i] > \
                        ohlc_renko["macd_sig_slope"][i]:
                    tickers_signal = "Buy"
                elif ohlc_renko["macd"][i] > ohlc_renko["macd_sig"][i] and \
                        ohlc_renko["macd_slope"][i] > ohlc_renko["macd_sig_slope"][i]:
                    tickers_signal = ""

            if tickers_signal == 'Sell' or tickers_signal == 'Buy':
                df_signal.append(tickers_signal)
            else:
                df_signal.append("Exit")

    ohlc_renko["signal"] = np.array(df_signal)
    ohlc_renko["ret"] = np.array(tickers_ret)
    return ohlc_renko


def run(df):
    candle_with_renko = merge_dfs(df)
    return identify_signal_return_renko_macd(df, candle_with_renko)


def main():
    df = ld.load_data()
    now_str = dt.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    df = ld.filter_df_by_interval(df, const.since, now_str)
    renko_macd = run(df)
    print(renko_macd)


if __name__ == '__main__':
    main()
