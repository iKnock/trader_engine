import numpy as np
from stocktrends import Renko
import statsmodels.api as sm
import transformer.indicators as indicator
import copy
import pandas as pd


def renko_dfr(data_f):
    "function to convert ohlc data into renko bricks"
    df = data_f.copy()
    df.reset_index(inplace=True)
    df = df.iloc[:, [0, 1, 2, 3, 4, 5]]
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df2 = Renko(df)
    df2.brick_size = max(0.5, round(indicator.atr(data_f, 120)[-1], 0))
    renko_df = pd.DataFrame(df2.get_ohlc_data())
    #renko_df = df2.get_ohlc_data()
    renko_df["bar_num"] = np.where(renko_df["uptrend"] == True, 1, np.where(renko_df["uptrend"] == False, -1, 0))
    for i in range(1, len(renko_df["bar_num"])):
        if renko_df["bar_num"][i] > 0 and renko_df["bar_num"][i - 1] > 0:
            renko_df["bar_num"][i] += renko_df["bar_num"][i - 1]
        elif renko_df["bar_num"][i] < 0 and renko_df["bar_num"][i - 1] < 0:
            renko_df["bar_num"][i] += renko_df["bar_num"][i - 1]
    renko_df.drop_duplicates(subset="date", keep="last", inplace=True)
    return renko_df


def obv(data_fr):
    """function to calculate On Balance Volume"""
    df = data_fr.copy()
    df['daily_ret'] = df['CLOSE'].pct_change()
    df['direction'] = np.where(df['daily_ret'] >= 0, 1, -1)
    df['direction'][0] = 0
    df['vol_adj'] = df['VOLUME'] * df['direction']
    df['obv'] = df['vol_adj'].cumsum()
    return df['obv']


def merge_dfs(df_intra_day):
    """Merging renko df with original ohlc df"""
    df = copy.deepcopy(df_intra_day)
    print("merging for ")
    renko = renko_dfr(df)
    renko.columns = ["Date", "open", "high", "low", "close", "uptrend", "bar_num"]
    df["Date"] = df.index
    ohlc_renko = df.merge(renko.loc[:, ["Date", "bar_num"]], how="outer", on="Date")
    ohlc_renko["bar_num"].fillna(method='ffill', inplace=True)
    ohlc_renko["obv"] = obv(ohlc_renko)
    ohlc_renko["obv_slope"] = indicator.slope(ohlc_renko["obv"], 5)
    return ohlc_renko


def identify_signal_return(df_intra, ohlc_renko):
    """Identifying signals and calculating daily return"""
    print("calculating daily returns for ")
    ohlc_intraday = copy.deepcopy(df_intra)

    tickers_signal = ""
    tickers_ret = []
    df_signal = []

    for i in range(len(ohlc_intraday)):
        if tickers_signal == "":
            tickers_ret.append(0)
            if ohlc_renko["bar_num"][i] >= 2 and ohlc_renko["obv_slope"][i] > 30:
                tickers_signal = "Buy"
            elif ohlc_renko["bar_num"][i] <= -2 and ohlc_renko["obv_slope"][i] < -30:
                tickers_signal = "Sell"

            if tickers_signal == 'Sell' or tickers_signal == 'Buy':
                df_signal.append(tickers_signal)
            else:
                df_signal.append("N/A")

        elif tickers_signal == "Buy":
            tickers_ret.append(
                (ohlc_renko["CLOSE"][i] / ohlc_renko["CLOSE"][i - 1]) - 1)
            if ohlc_renko["bar_num"][i] <= -2 and ohlc_renko["obv_slope"][i] < -30:
                tickers_signal = "Sell"
            elif ohlc_renko["bar_num"][i] < 2:
                tickers_signal = ""

            if tickers_signal == 'Sell' or tickers_signal == 'Buy':
                df_signal.append(tickers_signal)
            else:
                df_signal.append("N/A")

        elif tickers_signal == "Sell":
            tickers_ret.append(
                (ohlc_renko["CLOSE"][i - 1] / ohlc_renko["CLOSE"][i]) - 1)
            if ohlc_renko["bar_num"][i] >= 2 and ohlc_renko["obv_slope"][i] > 30:
                tickers_signal = "Buy"
            elif ohlc_renko["bar_num"][i] > -2:
                tickers_signal = ""

            if tickers_signal == 'Sell' or tickers_signal == 'Buy':
                df_signal.append(tickers_signal)
            else:
                df_signal.append("N/A")

    # df_signal.remove(df_signal[0])
    ohlc_renko["signal"] = np.array(df_signal)
    ohlc_renko["ret"] = np.array(tickers_ret)
    return ohlc_renko


def run(df):
    candle_with_renko = merge_dfs(df)
    ren_obv = identify_signal_return(df, candle_with_renko)
    return ren_obv
