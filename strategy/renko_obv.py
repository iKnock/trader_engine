import numpy as np
import transformer.indicators as indicator
import copy
import pandas as pd
import transformer.renko as rnk

pd.options.mode.chained_assignment = None  # default='warn'


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
    renko = rnk.renko_dfr(df)
    renko.columns = ["Date", "open", "high", "low", "close", "uptrend", "bar_num"]
    df["Date"] = df.index
    ohlc_renko = df.merge(renko.loc[:, ["Date", "bar_num"]], how="outer", on="Date")
    ohlc_renko["bar_num"].fillna(method='ffill', inplace=True)
    ohlc_renko["obv"] = obv(ohlc_renko)
    ohlc_renko["obv_slope"] = rnk.slope(ohlc_renko["obv"], 5)
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
                tickers_signal = ""  # exit

            if tickers_signal == 'Sell' or tickers_signal == 'Buy':
                df_signal.append(tickers_signal)
            else:
                df_signal.append("Exit")

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
                df_signal.append("Exit")

    ohlc_renko["signal"] = np.array(df_signal)
    ohlc_renko["ret"] = np.array(tickers_ret)
    return ohlc_renko


def run(df):
    candle_with_renko = merge_dfs(df)
    ren_obv = identify_signal_return(df, candle_with_renko)
    return ren_obv
