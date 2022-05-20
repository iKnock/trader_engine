# -*- coding: utf-8 -*-
"""
Created on Thu May 19 23:55:03 2022

@author: HSelato
"""

from pathlib import Path
import pandas as pd
import datetime as dt
import sys
import requests
import json


def read_csv_last_date():
    p = Path("./data/raw/Binance/BTC_euro_30m.csv")
    try:
        btc_df = pd.read_csv(p)
        df_candle = pd.DataFrame(btc_df)
        formated_date = pd.to_datetime(df_candle.tail(1).values[0, 0], unit='ms')
        date_str = dt.datetime.strptime(str(formated_date), '%Y-%m-%d %H:%M:%S').strftime(
            '%Y-%m-%d %H:%M:%S')
        return date_str
    except Exception:
        raise ValueError(' No File exist by this path')


def read_csv_df(csv_path):
    p = Path(csv_path)
    btc_df = pd.read_csv(p)
    df_candle = pd.DataFrame(btc_df)
    return format_candle_data(df_candle)


def format_candle_data(df):
    df_candle = df.copy()

    df_candle.columns = ["TIMESTAMP", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]

    df_candle = df_candle.set_index('TIMESTAMP')
    df_candle['DATE'] = pd.to_datetime(df_candle.index, utc=True, unit='ms').tz_convert('europe/rome')
    #df_candle['DATE'] = pd.to_datetime(df_candle.tail(1).values[0, 0], unit='ms')
    df_candle = df_candle.set_index('DATE')
    return df_candle


def run_query(host, sql_query):

    query_params = {'query': sql_query, 'fmt': 'json'}
    try:
        response = requests.get(host + '/exec', params=query_params)
        json_response = json.loads(response.text)
        print(json_response)
        return json_response
    except requests.exceptions.RequestException as e:
        print(f'Error: {e}', file=sys.stderr)



"""
 # create table
run_query("CREATE TABLE IF NOT EXISTS tbl_btc_5m_candle (insert_ts TIMESTAMP, DATE TIMESTAMP, OPEN DOUBLE, HIGH DOUBLE, LOW DOUBLE, CLOSE DOUBLE, VOLUME DOUBLE ")
 
 # insert row
run_query("INSERT INTO tbl_btc_5m_candle VALUES(now(), "+123456+")")      
"""