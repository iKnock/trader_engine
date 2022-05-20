# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:36:25 2022

@author: HSelato
"""

import os
from pathlib import Path

import sys
import csv
import requests
import ccxt
import utility.util as util
import pandas as pd

# -----------------------------------------------------------------------------

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(''))))
sys.path.append(root + '/codes/TRADER-ENGINE/trader_engine/csv')


# -----------------------------------------------------------------------------

def retry_fetch_ohlcv(exchange, max_retries, symbol, timeframe, since, limit):
    num_retries = 0
    try:
        num_retries += 1
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
        # print('Fetched', len(ohlcv), symbol, 'candles from', exchange.iso8601 (ohlcv[0][0]), 'to', exchange.iso8601 (ohlcv[-1][0]))
        return ohlcv
    except Exception:
        if num_retries > max_retries:
            raise  # Exception('Failed to fetch', timeframe, symbol, 'OHLCV in', max_retries, 'attempts')


def scrape_ohlcv(exchange, max_retries, symbol, timeframe, since, limit):
    earliest_timestamp = exchange.milliseconds()
    timeframe_duration_in_seconds = exchange.parse_timeframe(timeframe)
    timeframe_duration_in_ms = timeframe_duration_in_seconds * 1000
    timedelta = limit * timeframe_duration_in_ms
    all_ohlcv = []
    while True:
        fetch_since = earliest_timestamp - timedelta
        ohlcv = retry_fetch_ohlcv(exchange, max_retries, symbol, timeframe, fetch_since, limit)
        if len(ohlcv) > 0:
            # if we have reached the beginning of history
            print('*******************************************')
            print(len(ohlcv))
            print('*******************************************')
            if ohlcv[0][0] >= earliest_timestamp:
                break
            earliest_timestamp = ohlcv[0][0]
            all_ohlcv = ohlcv + all_ohlcv
            print(len(all_ohlcv), symbol, 'candles in total from', exchange.iso8601(all_ohlcv[0][0]), 'to',
                  exchange.iso8601(all_ohlcv[-1][0]))
            # if we have reached the checkpoint
            if fetch_since < since:
                break
        else:
            print(symbol + ' has no market data ')
            return all_ohlcv
    return all_ohlcv


def write_to_csv(filename, exchange, data, symbol, append):
    p = Path("./data/raw/", str(exchange))
    p.mkdir(parents=True, exist_ok=True)
    full_path = p / str(filename)

    with Path(full_path).open('w+', newline='') as output_file:
        csv_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerows(data)

    return full_path


def scrape_candles_to_csv(filename, exchange_id, max_retries, symbol, timeframe, since, limit, append):
    # instantiate the exchange by id
    exchange = getattr(ccxt, exchange_id)({
        'enableRateLimit': True,  # required by the Manual
    })
    # convert since from string to milliseconds integer if needed
    if isinstance(since, str):
        since = exchange.parse8601(since)
    # preload all markets from the exchange
    exchange.load_markets()
    # fetch all candles
    ohlcv = scrape_ohlcv(exchange, max_retries, symbol, timeframe, since, limit)
    if len(ohlcv) > 0:
        # save them to csv file
        full_path = write_to_csv(filename, exchange, ohlcv, symbol, append)

        import_csv_to_quest_db(full_path, append)

        print('Saved', len(ohlcv), 'candles from', exchange.iso8601(ohlcv[0][0]), 'to', exchange.iso8601(ohlcv[-1][0]),
              'to', filename)
        return ohlcv
    else:
        print("No data found for " + symbol + " at " + exchange_id)


def import_csv_to_quest_db(file_path, append):
    print('*******************************************')
    print(file_path)
    print('*******************************************')

    if append:
        table_name = os.path.basename(os.path.normpath(file_path))
        query = "DROP TABLE " + "'" + table_name + "'"
        host = 'http://localhost:9000'
        delete_resp = util.run_query(host, query)

        print('**********Delete Response******************')
        print(delete_resp)
        print('*******************************************')

    files = {
        'data': open(file_path, 'rb'),
    }

    response = requests.post('http://localhost:9000/imp', files=files)
    print('********import response********************')
    print(response)
    print('*******************************************')


def get_candles(file_name, exchange, max_retries, symbol, candle_size, since, limit, append):
    return scrape_candles_to_csv(file_name,
                                 exchange,
                                 max_retries,
                                 symbol,
                                 candle_size,
                                 since,
                                 limit,
                                 append)