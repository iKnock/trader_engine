# -*- coding: utf-8 -*-
"""
Created on Mon May 16 11:11:08 2022

@author: HSelato
"""
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import datetime as dt

import pandas as pd
import time

            
def GET_YAHOO_OHLCV(ticker, yperiod, interval):         
    ohlcv_data = {}        
    temp = yf.download(ticker, period=yperiod, interval=interval)
    #temp.dropna(how="any", inplace=True)
    temp.dropna(axis=0,how='any',inplace=True)#inplace true makes it substract from the data frame
    ohlcv_data[ticker] = temp
    return ohlcv_data    

def GET_BINANCE_OHLCV(ticker, exchange,ccxt, candle_size, since):
    """            
    '1h',limit=100
    """    
    msec = 1000
    minute = 60 * msec
    hold = 30
    
    exchange = ccxt.bitfinex({
        'rateLimit': 10000,
        'enableRateLimit': True,
        # 'verbose': True,
    })
    
    from_datetime = '2021-05-18 00:00:00'
    from_timestamp = exchange.parse8601(from_datetime)
    
    now = exchange.milliseconds()

# -----------------------------------------------------------------------------

    data = []
    
    while from_timestamp < now:

        try:
            #btc_usdt_ohlcv = pd.DataFrame(exchange.fetch_ohlcv('ADA/USDT', '15m', since=since, limit=1000))        
            
            print(exchange.milliseconds(), 'Fetching candles starting from', exchange.iso8601(from_timestamp))

            btc_usdt_ohlcv = exchange.fetch_ohlcv(ticker,candle_size,from_timestamp)    
                
            print(exchange.milliseconds(), 'Fetched', len(btc_usdt_ohlcv), 'candles')
            
            df_exchange_ohlcv = pd.DataFrame(btc_usdt_ohlcv)
            df_exchange_ohlcv.columns  = ["TIMESTAMP", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
            from_timestamp += len(df_exchange_ohlcv) * minute * 5
           # df_exchange_ohlcv['TIMESTAMP'] = pd.to_datetime(df_exchange_ohlcv['TIMESTAMP'], unit='ms')   
            data+=df_exchange_ohlcv
            
            return data
        except (ccxt.ExchangeError, ccxt.AuthenticationError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as error:
    
            print('Got an error', type(error).__name__, error.args, ', retrying in', hold, 'seconds...')
            time.sleep(hold)
    

def GET_ORDER_BOOK(exchange, ticker):
    orderbook_binance_btc_usdt = exchange.fetch_order_book(ticker) 
    
    bids_binance = orderbook_binance_btc_usdt['bids'] 
    asks_binanace = orderbook_binance_btc_usdt['asks']
    
    df_bid_binance = pd.DataFrame(bids_binance, columns=['price','qty']) 
    df_ask_binance = pd.DataFrame(asks_binanace, columns=['price','qty'])
    
    ticker_order_book = {}
    ticker_order_book['bid_binance'] = df_bid_binance
    ticker_order_book['ask_binance'] = df_ask_binance
    return ticker_order_book

#===============================Using crawler============================
def GET_CRYPTO_SUMMARY_PAGE(url, headers):
    page = requests.get(url, headers=headers)
    page_content = page.content       
    soup = BeautifulSoup(page_content, "html.parser")
    return soup

def CRYPTO_SUMMARY(soup):
    tbl = soup.find_all("table", {"class": "W(100%)"})   
    table_vals = {}
    for i in tbl:
        rows = i.find_all('tr')    
        index = 0
        for row in rows:                
            print(row.get_text(separator='|').split("|"))
            table_vals[index] =row.get_text(separator='|').split("|")
            index+=1            
    return table_vals
        
def FORMAT_CRYPTO_SUMMARY_DF(summary_table):
    crypto_summary_df = pd.DataFrame(summary_table.values())        
    #remove the last two col which are 52 week range and day chart which are none
    crypto_summary_df.drop(crypto_summary_df.columns[[-1,-2]], axis=1, inplace=True)
    
    new_header = crypto_summary_df.iloc[0] #grab the first row for the header
    crypto_summary_df = crypto_summary_df[1:] #take the data less the header row
    crypto_summary_df.columns = new_header #set the header row as the df header
    return crypto_summary_df