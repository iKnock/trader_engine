# -*- coding: utf-8 -*-
"""
Created on Thu May 19 23:55:03 2022

@author: HSelato
"""

from pathlib import Path
import pandas as pd
# import datetime as dt
from datetime import datetime as dt, timezone as tz

import sys
import requests
import json
import utility.constants as const
import numpy as np


def read_csv_last_date(exchange, file_name):
    p = Path("./data/raw/", str(exchange))
    full_path = p / str(file_name)
    try:
        btc_df = pd.read_csv(full_path)
        df_candle = pd.DataFrame(btc_df)
        formated_date = pd.to_datetime(df_candle.tail(1).values[0, 0], unit='ms')
        date_str = dt.strptime(str(formated_date), '%Y-%m-%d %H:%M:%S').strftime(
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
    # df_candle['DATE'] = pd.to_datetime(df_candle.tail(1).values[0, 0], unit='ms')
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


def calc_limit(since):
    now = dt.utcnow()  #
    # since = '2022-05-20 15:15:00'

    since_date = dt.strptime(since, '%Y-%m-%d %H:%M:%S')  # is on utc 00 timezone
    #    print(since_date.tzinfo)
    duration = now - since_date

    seconds_in_day = 24 * 60 * 60
    diff = divmod(duration.days * seconds_in_day + duration.seconds, 60)
    (0, 8)
    limit = None
    can_s = const.candle_size
    print(type(can_s))
    can_s = int(can_s.removesuffix(can_s[-1]))
    can_s = int(can_s)
    if const.candle_unit == "minute":
        limit = int(diff[0] / can_s)
        print("==========limit==========")
        print(limit)
    return limit
