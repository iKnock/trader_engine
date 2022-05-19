# -*- coding: utf-8 -*-
"""
Created on Mon May 16 12:55:16 2022

@author: HSelato
"""


#import ccxt.async_support as ccxt
#async def print_poloniex_ethbtc_ticker():
#    poloniex = ccxt.poloniex()
#    print(await poloniex.fetch_ticker('ETH/BTC'))

import ccxt
import asyncio
import pandas as pd
from dataminner import *

import sys
import requests
import json
from influx_line_protocol import Metric

import matplotlib.pyplot as plt
import datetime as dt
import numpy as np

from indicators import *
from strategy_performance_index import StrategyPerformanceIndex
from portofolio_rebalancing import Backtesting
from breakout import *

from asciichart import *


strategyPerfIndex = StrategyPerformanceIndex()
portofoRebala = Backtesting()  

start = dt.datetime.today()-dt.timedelta(3650)
end = dt.datetime.today()

def GET_YAHOO_HOURLY_OHLCV():
    tickers = ["MSFT","AAPL","GOOG"]
    yfinance_ohlcv={}
    for ticker in tickers:
        yfinance_ohlcv[ticker] = GET_YAHOO_OHLCV(ticker, '1y','1h')
    return yfinance_ohlcv    

def GET_YAHOO_OHLCV():
    tickers = ["MSFT","AAPL","GOOG"]
    yfinance_ohlcv={}
    for ticker in tickers:
        yfinance_ohlcv[ticker] = GET_YAHOO_OHLCV(ticker, '1mon','15m')
    return yfinance_ohlcv


def CRYPTO_SUMMERY():
    num_of_crypto = 200
    crypto_summary_url = "https://finance.yahoo.com/cryptocurrencies/?offset=0&count={}".format(num_of_crypto)
    headers = {"User-Agent": "Chrome/96.0.4664.110"}
    
    soup = GET_CRYPTO_SUMMARY_PAGE(crypto_summary_url, headers)
    summary_table_vals = CRYPTO_SUMMARY(soup)
    crypto_summery = FORMAT_CRYPTO_SUMMARY_DF(summary_table_vals)
    return crypto_summery

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
#==============================================================================
#==============================================================================
#==============================================================================

# Current time in nanoseconds
def current_timestamp():
    return int((dt.datetime.utcnow() - dt.datetime(1970, 1, 1)).total_seconds() * 1000) * 1000000

current_timestamp()             

def calc_and_add_indicators(DF):
    #binance_btc_5m_ohlcv = copy.deepcopy(DF)
    binance_btc_5m_ohlcv = btc_euro_5m;
    binance_btc_5m_ohlcv = CALC_APPEND_MACD(btc_euro_5m)
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
    return binance_btc_5m_ohlcv;



btc_euro_1h_with_indicators = pd.DataFrame(
    calc_and_add_indicators(btc_euro_5m))

df = breakout(ohlc_dict)

print("\n" + plot(df_5m_candle['CLOSE'][-40:],{'height': 15}))  # print the chart

#==============================================================================
#=============================Rendko data======================================
#==============================================================================

renko_data = CALC_APPEND_RENKO(btc_euro_5m.iloc[:,[0,1,2,3,4]],btc_euro_1h)

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
#==============================================================================
#==============================================================================
#==============================================================================

#===Strategy Performance KPI==
strategyPerfIndex.volatility(binance_btc_5m_ohlcv)
strategyPerfIndex.max_dd(binance_btc_5m_ohlcv,"five_minute_ret")
print("Sharpe = {}".format(strategyPerfIndex.sharpe(binance_btc_5m_ohlcv,0.03)))
print("Sortino = {}".format(strategyPerfIndex.sortino(binance_btc_5m_ohlcv,0.03)))

#visualization
fig, ax = plt.subplots()
plt.plot(df["ret"])
plt.title("BTC/USD hourly return")
plt.ylabel("cumulative return")
plt.xlabel("5Min")
ax.legend(["BTC/USD 5min_ret","CAGR", "sharpe", "sortino"])
#==============================================================================
#==============================================================================
#========================1 hour btc data with indicators=======================
#==============================================================================
#==============================================================================

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

#calculating overall strategy's KPIs


#==============================================================================
#==============================================================================
#==============================================================================
#=======================GET CANDLES AND DRAW THE CHART=========================
#==============================================================================
#==============================================================================

binance = ccxt.binance()
symbol = 'BTC/USDT'
timeframe = '1h'

def print_chart(exchange, symbol, timeframe):
    
    # each ohlcv candle is a list of [ timestamp, open, high, low, close, volume ]
    index = 4  # use close price from each ohlcv candle

    height = 15
    length = 80

    print("\n" + exchange.name + ' ' + symbol + ' ' + timeframe + ' chart:')

    # get a list of ohlcv candles
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe)

    # get the ohlCv (closing price, index == 4)
    series = [x[index] for x in ohlcv]
    # print the chart
    print("\n" + plot(series[-length:],{'height': height}))  # print the chart

    last = ohlcv[len(ohlcv) - 1][index]  # last closing price
    return last

last = print_chart(binance, symbol, timeframe)

# print the chart
#==============================================================================
#==============================================================================
#========================query quest db========================================
#==============================================================================
#==============================================================================

host = 'http://localhost:9000'

def run_query(sql_query):
    query_params = {'query': sql_query, 'fmt' : 'json'}
    try:
        response = requests.get(host + '/exec', params=query_params)
        json_response = json.loads(response.text)
        print(json_response)
    except requests.exceptions.RequestException as e:
        print(f'Error: {e}', file=sys.stderr)
        
 # create table
run_query("CREATE TABLE IF NOT EXISTS tbl_btc_5m_candle (insert_ts TIMESTAMP, DATE TIMESTAMP, OPEN DOUBLE, HIGH DOUBLE, LOW DOUBLE, CLOSE DOUBLE, VOLUME DOUBLE ")
 
 # insert row
run_query("INSERT INTO tbl_btc_5m_candle VALUES(now(), "+123456+")")          
