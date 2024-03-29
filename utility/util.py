from pathlib import Path
import pandas as pd
from datetime import datetime as dt, timezone as tz

import sys
import requests
import json
import utility.constants as const


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
    df_candle['DATE'] = pd.to_datetime(df_candle.index, utc=True, unit='ms')
    df_candle = df_candle.set_index('DATE')
    return df_candle


def convert_df_timezone(data_frame):
    df = data_frame.copy()
    df['DATE'] = pd.to_datetime(df.index, utc=True, unit='ms').tz_convert('europe/rome')
    return df.set_index('DATE')


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
    now = dt.now()# is on utc 00 timezone
    since_date = dt.strptime(since, '%Y-%m-%d %H:%M:%S')  # is on utc +2 timezone

    duration = now - since_date

    seconds_in_day = 24 * 60 * 60
    diff = divmod(duration.days * seconds_in_day + duration.seconds, 60)
    limit = None
    can_s = const.candle_size
    print(type(can_s))
    can_s = int(can_s[:-1])
    can_s = int(can_s)
    if const.candle_unit == "min":
        limit = int(diff[0] / can_s)
    elif const.candle_unit == 'hr':
        limit = int(diff[0]/60)/can_s
    # else convert unit of diff[0] to candle_unit

    print("==========limit==========")
    print(limit)
    return limit
