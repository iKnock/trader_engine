# -*- coding: utf-8 -*-
"""
Created on Tue May 17 16:43:09 2022

@author: HSelato
"""
import numpy as np
import pandas as pd
import copy


def CAGR(DF):
    "function to calculate the Cumulative Annual Growth Rate of a trading strategy"
    df = copy.deepcopy(DF)
    df["return"] = DF["CLOSE"].pct_change()
    df["cum_return"] = (1 + df["return"]).cumprod()
    n = len(df) / 252
    CAGR = (df["cum_return"]) ** (1 / n) - 1
    return CAGR


def volatility(DF):
    "function to calculate annualized volatility of a trading strategy"
    df = copy.deepcopy(DF)
    df["return"] = df["CLOSE"].pct_change()
    vol = df["return"].std() * np.sqrt(252)
    return vol


def sharpe(DF, rf):
    "function to calculate Sharpe Ratio of a trading strategy"
    df = copy.deepcopy(DF)
    return (CAGR(df) - rf) / volatility(df)


def sortino(DF, rf):
    "function to calculate Sortino Ratio of a trading strategy"
    df = copy.deepcopy(DF)
    df["return"] = df["CLOSE"].pct_change()
    neg_return = np.where(df["return"] > 0, 0, df["return"])
    neg_vol = pd.Series(neg_return[neg_return != 0]).std() * np.sqrt(252)
    return (CAGR(df) - rf) / neg_vol


def max_dd(DF, return_key):
    "function to calculate max drawdown"
    df = copy.deepcopy(DF)
    df["cum_return"] = (1 + df[return_key]).cumprod()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    df["drawdown_pct"] = df["drawdown"] / df["cum_roll_max"]
    max_dd = df["drawdown_pct"].max()
    return max_dd


def RETURN_FOR_PERIOD(DF, period):
    # calculating monthly return for each stock and consolidating return info by stock in a separate dataframe
    ohlc_dict = copy.deepcopy(DF)
    return_df = pd.DataFrame()
    print("calculating " + period + " return")
    ohlc_dict[period + "_ret"] = ohlc_dict["CLOSE"].pct_change()
    return_df = ohlc_dict[period + "_ret"]
    return_df.dropna(inplace=True)
    return return_df
