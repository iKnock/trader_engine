# -*- coding: utf-8 -*-
"""
Created on Mon May 16 12:55:16 2022

@author: HSelato
"""

import ccxt
import pandas as pd
from dataminner import DataMinner
from indicators import Indicators
from strategy_performance_index import StrategyPerformanceIndex
from portofolio_rebalancing import Backtesting
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np


data_minner = DataMinner()
indicator = Indicators()
strategyPerfIndex = StrategyPerformanceIndex()
portofoRebala = Backtesting()  

start = dt.datetime.today()-dt.timedelta(3650)
end = dt.datetime.today()

def GET_YAHOO_HOURLY_OHLCV():
    tickers = ["MSFT","AAPL","GOOG"]
    yfinance_ohlcv={}
    for ticker in tickers:
        yfinance_ohlcv[ticker] = data_minner.GET_YAHOO_OHLCV(ticker, '1y','1h')
    return yfinance_ohlcv    

def GET_YAHOO_OHLCV():
    tickers = ["MSFT","AAPL","GOOG"]
    yfinance_ohlcv={}
    for ticker in tickers:
        yfinance_ohlcv[ticker] = data_minner.GET_YAHOO_OHLCV(ticker, '1mon','15m')
    return yfinance_ohlcv


def CRYPTO_SUMMERY():
    num_of_crypto = 200
    crypto_summary_url = "https://finance.yahoo.com/cryptocurrencies/?offset=0&count={}".format(num_of_crypto)
    headers = {"User-Agent": "Chrome/96.0.4664.110"}
    
    soup = data_minner.GET_CRYPTO_SUMMARY_PAGE(crypto_summary_url, headers)
    summary_table_vals = data_minner.CRYPTO_SUMMARY(soup)
    crypto_summery = data_minner.FORMAT_CRYPTO_SUMMARY_DF(summary_table_vals)
    return crypto_summery

def CALC_APPEND_MACD(DF):
    df = DF.copy()
    macd=pd.DataFrame(indicator.MACD(df))
    df = df.assign(MACD=pd.Series(macd["macd"]).values,Signal=pd.Series(macd["signal"]).values)
    return df

def CALC_APPEND_ATR(DF):
    df = DF.copy()
    atr=pd.DataFrame(indicator.ATR(df))
    df = df.assign(ATR=pd.Series(atr['ATR']).values)
    return df

def CALC_APPEND_BB(DF):
    df = DF.copy()
    boll_band=pd.DataFrame(indicator.Boll_Band(df))
    df = df.assign(MB=pd.Series(boll_band['MB']).values,
                                 UB=pd.Series(boll_band['UB']).values,
                                 LB=pd.Series(boll_band['LB']).values,
                                 BB_Width=pd.Series(boll_band['BB_Width']).values)
    return df

def CALC_APPEND_ADX(DF):
    df = DF.copy()
    atr=pd.DataFrame(indicator.ADX(df))
    df = df.assign(ADX=pd.Series(atr['ADX']).values)
    return df

def CALC_APPEND_RSI(DF):
    df = DF.copy()
    atr=pd.DataFrame(indicator.RSI(df))
    df = df.assign(RSI=pd.Series(atr['rsi']).values)
    return df

def CALC_APPEND_RENKO(DF, binance_btc_ohlcv):
    df = DF.copy()
    rendo_data=pd.DataFrame(
        indicator.renko_DF(df, binance_btc_ohlcv)
        )    
    return rendo_data

#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
binance_btc_order_book = data_minner.GET_ORDER_BOOK(
    ccxt.binance({'verbose': True}), 
    "BTC/USDT")

yahoo_ohlcv = GET_YAHOO_OHLCV()
yahoo_ohlcv_hourly = GET_YAHOO_HOURLY_OHLCV()

crypto_summery_crawler = CRYPTO_SUMMERY()

#==============================================================================
#==============================================================================
#========================5 Min btc data with indicators=======================
#==============================================================================
#==============================================================================

binance_btc_5m_ohlcv = data_minner.GET_BINANCE_OHLCV(
    "BTC/USDT", 
    ccxt.binance({'verbose': True}),
    '5m',
    200)

binance_btc_5m_ohlcv['TIMESTAMP'] = pd.to_datetime(binance_btc_5m_ohlcv['TIMESTAMP'], unit='ms')
binance_btc_5m_ohlcv = binance_btc_5m_ohlcv.iloc[::-1]
binance_btc_5m_ohlcv.reset_index(inplace=True)
binance_btc_5m_ohlcv.drop("index",axis=1,inplace=True)

binance_btc_5m_ohlcv = CALC_APPEND_MACD(binance_btc_5m_ohlcv)
binance_btc_5m_ohlcv = CALC_APPEND_ATR(binance_btc_5m_ohlcv)
binance_btc_5m_ohlcv = CALC_APPEND_BB(binance_btc_5m_ohlcv)
binance_btc_5m_ohlcv = CALC_APPEND_ADX(binance_btc_5m_ohlcv)
binance_btc_5m_ohlcv = CALC_APPEND_RSI(binance_btc_5m_ohlcv)

binance_btc_5m_ohlcv["roll_max_cp"] = binance_btc_5m_ohlcv["HIGH"].rolling(20).max()
binance_btc_5m_ohlcv["roll_min_cp"] = binance_btc_5m_ohlcv["LOW"].rolling(20).min()
binance_btc_5m_ohlcv["roll_max_vol"] = binance_btc_5m_ohlcv["VOLUME"].rolling(20).max()

binance_btc_5m_ohlcv['five_minute_ret'] = strategyPerfIndex.RETURN_FOR_PERIOD(
    binance_btc_5m_ohlcv,"five_minute")

binance_btc_5m_ohlcv['CAGR']=strategyPerfIndex.CAGR(binance_btc_5m_ohlcv)
binance_btc_5m_ohlcv['sharpe']=strategyPerfIndex.sharpe(binance_btc_5m_ohlcv,0.03)
binance_btc_5m_ohlcv['sortino']=strategyPerfIndex.sortino(binance_btc_5m_ohlcv,0.03)

binance_btc_5m_ohlcv = breakout(binance_btc_5m_ohlcv)

def breakout(DF):
    tickers_signal = {}
    tickers_ret = {}
    ohlc_dict = binance_btc_5m_ohlcv.copy()
    print("calculating returns ")
    for i in range(1,len(ohlc_dict)):
        if tickers_signal == "":
            tickers_ret.append(0)
            if ohlc_dict["HIGH"][i]>=ohlc_dict["roll_max_cp"][i] and \
               ohlc_dict["VOLUME"][i]>1.5*ohlc_dict["roll_max_vol"][i-1]:
                tickers_signal = "Buy"
            elif ohlc_dict["LOW"][i]<=ohlc_dict["roll_min_cp"][i] and \
               ohlc_dict["VOLUME"][i]>1.5*ohlc_dict["roll_max_vol"][i-1]:
                tickers_signal = "Sell"
        
        elif tickers_signal == "Buy":
            if ohlc_dict["LOW"][i]<ohlc_dict["CLOSE"][i-1] - ohlc_dict["ATR"][i-1]:
                tickers_signal = ""
                tickers_ret.append(((ohlc_dict["CLOSE"][i-1] - ohlc_dict["ATR"][i-1])/ohlc_dict["Close"][i-1])-1)
            elif ohlc_dict["LOW"][i]<=ohlc_dict["roll_min_cp"][i] and \
               ohlc_dict["VOLUME"][i]>1.5*ohlc_dict["roll_max_vol"][i-1]:
                tickers_signal = "Sell"
                tickers_ret.append((ohlc_dict["CLOSE"][i]/ohlc_dict["CLOSE"][i-1])-1)
            else:
                tickers_ret.append((ohlc_dict["CLOSE"][i]/ohlc_dict["CLOSE"][i-1])-1)
                
        elif tickers_signal == "Sell":
            if ohlc_dict["HIGH"][i]>ohlc_dict["CLOSE"][i-1] + ohlc_dict["ATR"][i-1]:
                tickers_signal = ""
                tickers_ret.append((ohlc_dict["CLOSE"][i-1]/(ohlc_dict["CLOSE"][i-1] + ohlc_dict["ATR"][i-1]))-1)
            elif ohlc_dict["HIGH"][i]>=ohlc_dict["roll_max_cp"][i] and \
               ohlc_dict["VOLUME"][i]>1.5*ohlc_dict["roll_max_vol"][i-1]:
                tickers_signal = "Buy"
                tickers_ret.append((ohlc_dict["CLOSE"][i-1]/ohlc_dict["CLOSE"][i])-1)
            else:
                tickers_ret.append((ohlc_dict["CLOSE"][i-1]/ohlc_dict["CLOSE"][i])-1)
                
    ohlc_dict["ret"] = np.array(tickers_ret)
    return ohlc_dict




#===Strategy Performance KPI==
strategyPerfIndex.volatility(binance_btc_5m_ohlcv)
strategyPerfIndex.max_dd(binance_btc_5m_ohlcv,"five_minute_ret")
print("Sharpe = {}".format(strategyPerfIndex.sharpe(binance_btc_5m_ohlcv,0.03)))
print("Sortino = {}".format(strategyPerfIndex.sortino(binance_btc_5m_ohlcv,0.03)))

#visualization
fig, ax = plt.subplots()
plt.plot(binance_btc_5m_ohlcv["five_minute_ret"])
plt.plot(binance_btc_5m_ohlcv["CAGR"])
plt.plot(binance_btc_5m_ohlcv["sharpe"])
plt.plot(binance_btc_5m_ohlcv["sortino"])
plt.title("BTC/USD hourly return")
plt.ylabel("cumulative return")
plt.xlabel("5Min")
ax.legend(["BTC/USD 5min_ret","CAGR", "sharpe", "sortino"])
#==============================================================================
#==============================================================================
#========================1 hour btc data with indicators=======================
#==============================================================================
#==============================================================================

binance_btc_one_hour_ohlcv = data_minner.GET_BINANCE_OHLCV(
    "BTC/USDT", 
    ccxt.binance({'verbose': True}),
    '1h',
    1000)

binance_btc_one_hour_ohlcv['TIMESTAMP'] = pd.to_datetime(binance_btc_one_hour_ohlcv['TIMESTAMP'], unit='ms')
binance_btc_one_hour_ohlcv = binance_btc_one_hour_ohlcv.iloc[::-1]#reverse the df
binance_btc_one_hour_ohlcv.reset_index(inplace=True)
binance_btc_one_hour_ohlcv.drop("index",axis=1,inplace=True)#drop the old index


binance_btc_one_hour_ohlcv = CALC_APPEND_MACD(binance_btc_one_hour_ohlcv)
binance_btc_one_hour_ohlcv = CALC_APPEND_ATR(binance_btc_one_hour_ohlcv)
binance_btc_one_hour_ohlcv = CALC_APPEND_BB(binance_btc_one_hour_ohlcv)
binance_btc_one_hour_ohlcv = CALC_APPEND_ADX(binance_btc_one_hour_ohlcv)
binance_btc_one_hour_ohlcv = CALC_APPEND_RSI(binance_btc_one_hour_ohlcv)

binance_btc_one_hour_ohlcv['hourly_ret'] = strategyPerfIndex.RETURN_FOR_PERIOD(
    binance_btc_one_hour_ohlcv,"hourly")

binance_btc_one_hour_ohlcv['CAGR']=strategyPerfIndex.CAGR(binance_btc_one_hour_ohlcv)
binance_btc_one_hour_ohlcv['sharpe']=strategyPerfIndex.sharpe(binance_btc_one_hour_ohlcv,0.03)
binance_btc_one_hour_ohlcv['sortino']=strategyPerfIndex.sortino(binance_btc_one_hour_ohlcv,0.03)

#===============Strategy Performance KPI====================
strategyPerfIndex.volatility(binance_btc_one_hour_ohlcv)
strategyPerfIndex.max_dd(binance_btc_one_hour_ohlcv,"hourly_ret")
print("Sharpe = {}".format(strategyPerfIndex.sharpe(binance_btc_one_hour_ohlcv,0.03)))
print("Sortino = {}".format(strategyPerfIndex.sortino(binance_btc_one_hour_ohlcv,0.03)))


#visualization
fig, ax = plt.subplots()
plt.plot(binance_btc_one_hour_ohlcv["hourly_ret"])
plt.title("BTC/USD hourly return")
plt.ylabel("cumulative return")
plt.xlabel("months")
ax.legend(["BTC/USD hourly return","Index Return"])
#==============================================================================
#==============================================================================
#========================1 month btc data with indicators=======================
#==============================================================================
#==============================================================================

binance_btc_one_month_ohlcv = data_minner.GET_BINANCE_OHLCV(
    "BTC/USDT", 
    ccxt.binance({'verbose': True}),
    '1M',
    7000)

binance_btc_one_month_ohlcv['TIMESTAMP'] = pd.to_datetime(binance_btc_one_month_ohlcv['TIMESTAMP'], unit='ms')
binance_btc_one_month_ohlcv = binance_btc_one_month_ohlcv.iloc[::-1]
binance_btc_one_month_ohlcv.reset_index(inplace=True)
binance_btc_one_month_ohlcv.drop("index",axis=1,inplace=True)

binance_btc_one_month_ohlcv = CALC_APPEND_MACD(binance_btc_one_month_ohlcv)
binance_btc_one_month_ohlcv = CALC_APPEND_ATR(binance_btc_one_month_ohlcv)
binance_btc_one_month_ohlcv = CALC_APPEND_BB(binance_btc_one_month_ohlcv)
binance_btc_one_month_ohlcv = CALC_APPEND_ADX(binance_btc_one_month_ohlcv)
binance_btc_one_month_ohlcv = CALC_APPEND_RSI(binance_btc_one_month_ohlcv)

binance_btc_one_month_ohlcv["monthly_ret"] = strategyPerfIndex.RETURN_FOR_PERIOD(
    binance_btc_one_month_ohlcv,"monthly")

binance_btc_one_month_ohlcv['CAGR']=strategyPerfIndex.CAGR(binance_btc_one_month_ohlcv)
binance_btc_one_month_ohlcv['sharpe']=strategyPerfIndex.sharpe(binance_btc_one_month_ohlcv,0.03)
binance_btc_one_month_ohlcv['sortino']=strategyPerfIndex.sortino(binance_btc_one_month_ohlcv,0.03)

#===============Strategy Performance KPI====================
strategyPerfIndex.volatility(binance_btc_one_month_ohlcv)
strategyPerfIndex.max_dd(binance_btc_one_month_ohlcv,"monthly_ret")
print("Sharpe = {}".format(strategyPerfIndex.sharpe(binance_btc_one_month_ohlcv,0.03)))
print("Sortino = {}".format(strategyPerfIndex.sortino(binance_btc_one_month_ohlcv,0.03)))

#visualization
fig, ax = plt.subplots()
plt.plot(binance_btc_one_month_ohlcv["monthly_ret"])
plt.title("BTC/USD monthly return")
plt.ylabel("cumulative return")
plt.xlabel("months")
ax.legend(["BTC/USD monthly return","one hour Return","five min"])
#==============================================================================
#=============================Rendko data======================================
#==============================================================================
renko_data = CALC_APPEND_RENKO(binance_btc_5m_ohlcv.iloc[:,[0,1,2,3,4]],binance_btc_one_hour_ohlcv)

def convertRenkoToNumeric():
    renko_data['open'] = renko_data['open'].astype('float')
    renko_data['high'] = renko_data['high'].astype('float')
    renko_data['low'] = renko_data['low'].astype('float')
    renko_data['close'] = renko_data['close'].astype('float')
    renko_data['uptrend'] = renko_data['uptrend'].astype('float')
    return renko_data

#visualization
fig, ax = plt.subplots()
plt.plot(renko_data["close"])
plt.title("BTC/USD monthly return")
plt.ylabel("cumulative return")
plt.xlabel("months")
ax.legend(["BTC/USD monthly return","one hour Return","five min"])
#==============================================================================
#==============================================================================
#==============================================================================
#calculating overall strategy's KPIs
