# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:31:36 2022

@author: HSelato
"""

import ccxt
from dataminner import DataMinner

data_minner = DataMinner()

def GET_YAHOO_OHLCV():
    tickers = ["MSFT","AAPL","GOOG"]
    yfinance_ohlcv={}
    for ticker in tickers:
        yfinance_ohlcv[ticker] = data_minner.GET_YAHOO_OHLCV(ticker, '1mon','15m')
    return yfinance_ohlcv

def GET_BINANCE_BTC_OHLCV():
    ticker = "BTC/USDT"
    binance = ccxt.binance({'verbose': True})
    binance_btc_ohlcv = data_minner.GET_BINANCE_OHLCV(ticker, binance,'1h',200)
    return binance_btc_ohlcv

def GET_BINANCE_BTC_ORDER_BOOK():
    ticker = "BTC/USDT"
    binance = ccxt.binance({'verbose': True})
    binance_btc_order_book = data_minner.GET_ORDER_BOOK(binance, ticker)
    return binance_btc_order_book

def CRYPTO_SUMMERY():
    num_of_crypto = 200
    crypto_summary_url = "https://finance.yahoo.com/cryptocurrencies/?offset=0&count={}".format(num_of_crypto)
    headers = {"User-Agent": "Chrome/96.0.4664.110"}
    
    soup = data_minner.GET_CRYPTO_SUMMARY_PAGE(crypto_summary_url, headers)
    summary_table_vals = data_minner.CRYPTO_SUMMARY(soup)
    crypto_summery = data_minner.FORMAT_CRYPTO_SUMMARY_DF(summary_table_vals)
    return crypto_summery

yahoo_ohlcv = GET_YAHOO_OHLCV()
binance_btc_ohlcv = GET_BINANCE_BTC_OHLCV()
binance_btc_order_book = GET_BINANCE_BTC_ORDER_BOOK()
crypto_summery_crawler = CRYPTO_SUMMERY()