# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:45:52 2022

@author: HSelato
"""

import numpy as np
from stocktrends import Renko

def MACD(DF, a=12 ,b=26, c=9):
    """function to calculate MACD
       typical values a(fast moving average) = 12; 
                      b(slow moving average) =26; 
                      c(signal line ma window) =9"""
    df = DF.copy()
    df["ma_fast"] = df["CLOSE"].ewm(span=a, min_periods=a).mean()
    df["ma_slow"] = df["CLOSE"].ewm(span=b, min_periods=b).mean()
    df["macd"] = df["ma_fast"] - df["ma_slow"]
    df["signal"] = df["macd"].ewm(span=c, min_periods=c).mean()
    return df.loc[:,["macd","signal"]]# : means all and loc accept the rows and col as param

def ATR(DF, n=14):
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    df["H-L"] = df["HIGH"] - df["LOW"]
    df["H-PC"] = abs(df["HIGH"] - df["CLOSE"].shift(1))
    df["L-PC"] = abs(df["LOW"] - df["CLOSE"].shift(1))
    df["TR"] = df[["H-L","H-PC","L-PC"]].max(axis=1, skipna=False)
    df["ATR"] = df["TR"].ewm(com=n, min_periods=n).mean()
    return df["ATR"]

def Boll_Band(DF, n=14):
    "function to calculate Bollinger Band"
    df = DF.copy()
    df["MB"] = df["CLOSE"].rolling(n).mean()
    df["UB"] = df["MB"] + 2*df["CLOSE"].rolling(n).std(ddof=0)#ddof is the degree of freedom
    df["LB"] = df["MB"] - 2*df["CLOSE"].rolling(n).std(ddof=0)
    df["BB_Width"] = df["UB"] - df["LB"]
    return df[["MB","UB","LB","BB_Width"]]

def ADX(DF, n=20):
    "function to calculate ADX"
    df = DF.copy()
    df["ATR"] = ATR(DF, n)#for the formulas used refer trading view doc
    df["upmove"] = df["HIGH"] - df["HIGH"].shift(1)
    df["downmove"] = df["LOW"].shift(1) - df["LOW"]
    df["+dm"] = np.where((df["upmove"]>df["downmove"]) & (df["upmove"] >0), df["upmove"], 0)
    df["-dm"] = np.where((df["downmove"]>df["upmove"]) & (df["downmove"] >0), df["downmove"], 0)
    df["+di"] = 100 * (df["+dm"]/df["ATR"]).ewm(alpha=1/n, min_periods=n).mean()
    df["-di"] = 100 * (df["-dm"]/df["ATR"]).ewm(alpha=1/n, min_periods=n).mean()
    df["ADX"] = 100* abs((df["+di"] - df["-di"])/(df["+di"] + df["-di"])).ewm(alpha=1/n, min_periods=n).mean()
    return df["ADX"]

def RSI(DF, n=14):
    "function to calculate RSI"
    df = DF.copy()
    df["change"] = df["CLOSE"] - df["CLOSE"].shift(1)
    df["gain"] = np.where(df["change"]>=0, df["change"], 0)#numpy .where is like if else
    df["loss"] = np.where(df["change"]<0, -1*df["change"], 0)
    df["avgGain"] = df["gain"].ewm(alpha=1/n, min_periods=n).mean()
    df["avgLoss"] = df["loss"].ewm(alpha=1/n, min_periods=n).mean()
    df["rs"] = df["avgGain"]/df["avgLoss"]
    df["rsi"] = 100 - (100/ (1 + df["rs"]))
    return df["rsi"]

def renko_DF(DF, hourly_df):
    "function to convert ohlc data into renko bricks"
    df = DF.copy()
    #df.reset_index(inplace=True)
   # df.drop("VOLUME",axis=1,inplace=True)#Axis=1 signify Close is a column and inplace=True in this var not a copy        
    df.columns = ["date","open","high","low","close"]        
    df2 = Renko(df)
    #df2.brick_size = 3*round(self.ATR(hourly_df,120).iloc[-1],0)#iloc[-1] give the last value
    df2.brick_size = 4
    renko_df = df2.get_ohlc_data() #if using older version of the library please use get_bricks() instead
    return renko_df