# -*- coding: utf-8 -*-
"""
Created on Sun May 15 15:32:54 2022

@author: HSelato
"""

"""import ccxt.async_support as ccxt # link against the asynchronous version of ccxt"""
import ccxt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from stocktrends import Renko


# print a list of all available exchange classes
exchange = ccxt.exchanges

binance = ccxt.binance({'verbose': True})

#btc_ticker = binance.fetch_ticker('BTC/USDT')
#df_btc_ticker = pd.DataFrame(btc_ticker)

def TIKER_OHLCV(ticker):
    btc_usdt_ohlcv = binance.fetch_ohlcv(ticker,'1h',limit=100)
    df_binance_ohlcv = pd.DataFrame(btc_usdt_ohlcv,columns={"TIMESTAMP", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"})
    return df_binance_ohlcv


def TIKER_ORDER_BOOK(ticker):
    orderbook_binance_btc_usdt = binance.fetch_order_book(ticker) 
    
    bids_binance = orderbook_binance_btc_usdt['bids'] 
    asks_binanace = orderbook_binance_btc_usdt['asks']
    
    df_bid_binance = pd.DataFrame(bids_binance, columns=['price','qty']) 
    df_ask_binance = pd.DataFrame(asks_binanace, columns=['price','qty'])
    
    ticker_order_book = {}
    ticker_order_book['bid_binance'] = df_bid_binance
    ticker_order_book['ask_binance'] = df_ask_binance
    return ticker_order_book

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
    df.reset_index(inplace=True)
    df.drop("CLOSE",axis=1,inplace=True)#Axis=1 signify Close is a column and inplace=True in this var not a copy
    df.columns = ["date","open","high","low","close","volume"]
    df2 = Renko(df)
    df2.brick_size = 3*round(ATR(hourly_df,120).iloc[-1],0)#iloc[-1] give the last value
    renko_df = df2.get_ohlc_data() #if using older version of the library please use get_bricks() instead
    return renko_df

ticker = 'BTC/USDT' #LUNA-USD BTC/USDT
btc_ohlcv=TIKER_OHLCV(ticker)
binance_btc_order_book = TIKER_ORDER_BOOK(ticker)


btc_ohlcv.dropna(axis=0,how='any',inplace=True)#inplace true makes it substract from the data frame

macd=pd.DataFrame(MACD(btc_ohlcv))
btc_ohlcv = btc_ohlcv.assign(MACD=pd.Series(macd["macd"]).values,Signal=pd.Series(macd["signal"]).values)

atr=pd.DataFrame(ATR(btc_ohlcv))
btc_ohlcv = btc_ohlcv.assign(ATR=pd.Series(atr['ATR']).values)


boll_band=pd.DataFrame(Boll_Band(btc_ohlcv))
btc_ohlcv = btc_ohlcv.assign(MB=pd.Series(boll_band['MB']).values,
                             UB=pd.Series(boll_band['UB']).values,
                             LB=pd.Series(boll_band['LB']).values,
                             BB_Width=pd.Series(boll_band['BB_Width']).values)



plt.plot(boll_band)



# Mean, Median, Standard Deviation
mean = btc_ohlcv.mean() #calculates mean stock price for each stock
median = btc_ohlcv.median() #calculates median stock price for each stock
std = btc_ohlcv.std() #calculates standard deviation of stock price for each stock

#Note: quantitative finacial ananlysis relays on asset return not on asset value.

# Return calculation
daily_return = btc_ohlcv.pct_change() #Creates dataframe with daily return for each stock
daily_return = btc_ohlcv/btc_ohlcv.shift(1) - 1 #Performs same operation as above

daily_return.mean(axis=1) #prints mean daily return for each stock
daily_return.std() #prints standard deviation of daily returns for each stock
    
daily_return.rolling(window=10).mean()
daily_return.rolling(window=10).std()
daily_return.rolling(window=10).max()
daily_return.rolling(window=10).sum()

# Rolling operations (exponentialy weighted)
df1 = daily_return.ewm(com=10, min_periods=10).mean()
df2= daily_return.ewm(com=10, min_periods=10).std()


#familiarizing with data
btc_ohlcv.head(8) #prints the first 8 rows of the dataframe
btc_ohlcv.tail() #prints the last 5 rows of the dataframe
btc_ohlcv.describe() #performs summary statistic of each numerical columns in the datafrme

# Pyplot demo
fig, ax = plt.subplots()
plt.style.available
plt.style.use('ggplot')
ax.set(title="Daily return on tech stocks", xlabel="Tech Stocks", ylabel = "Daily Returns")
plt.bar(daily_return.columns,daily_return.mean(),color=["red","blue","green","orange"])
plt.plot(btc_ohlcv["CLOSE"])
plt.plot(btc_ohlcv["VOLUME"])