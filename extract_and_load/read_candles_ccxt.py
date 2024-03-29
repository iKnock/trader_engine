import os
from pathlib import Path
import sys
import csv
import requests
import utility.util as util
import utility.constants as const
from datetime import datetime as dt, timezone as tz, timedelta as td

# -----------------------------------------------------------------------------

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(''))))
sys.path.append(root + '/codes/TRADER-ENGINE/trader_engine')


# -----------------------------------------------------------------------------

def retry_fetch_ohlcv(exchange, max_retries, symbol, timeframe, since, limit):
    num_retries = 0
    try:
        num_retries += 1
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        # print('Fetched', len(ohlcv), symbol, 'candles from', exchange.iso8601 (ohlcv[0][0]), 'to', exchange.iso8601 (ohlcv[-1][0]))
        return ohlcv
    except Exception:
        if num_retries > max_retries:
            raise  # Exception('Failed to fetch', timeframe, symbol, 'OHLCV in', max_retries, 'attempts')


def scrape_ohlcv(exchange, max_retries, symbol, timeframe, since, limit):
    now = dt.now()
    opti_limit = 1000
    day_unit_val = 0
    candle_size = int(timeframe[:-1])

    if const.candle_unit == 'hr':
        day_unit_val = 24
    elif const.candle_unit == 'min':
        day_unit_val = 1440

    dur_in_days = (opti_limit * candle_size) / day_unit_val

    fetc_since = now - td(dur_in_days)

    # convert since from string to milliseconds integer if needed
    fet_sin_str = fetc_since.strftime("%Y-%m-%d %H:%M:%S")
    fetc_since = exchange.parse8601(fet_sin_str)

    all_ohlcv = []
    count = 0
    while True:
        # print(dt.fromtimestamp(fetc_since / 1e3))
        ohlcv = retry_fetch_ohlcv(exchange, max_retries, symbol, timeframe, fetc_since, limit)

        if len(ohlcv) > 0:
            dur_in_millisec = dur_in_days * 24 * 60 * 60 * 1000
            fetc_since = int(fetc_since - dur_in_millisec)

            all_ohlcv = ohlcv + all_ohlcv
            print(len(all_ohlcv), symbol, 'candles in total from', exchange.iso8601(all_ohlcv[0][0]), 'to',
                  exchange.iso8601(all_ohlcv[-1][0]))
            # if we have reached the checkpoint
            if count == 1:
                break
            if since >= fetc_since:
                fetc_since = since
                count += 1
        else:
            print(symbol + ' has no market data ')
            return all_ohlcv
    return all_ohlcv


# def scrape_ohlcv(exchange, max_retries, symbol, timeframe, since, limit):
#     earliest_timestamp = exchange.milliseconds()
#     timeframe_duration_in_seconds = exchange.parse_timeframe(timeframe)
#     timeframe_duration_in_ms = timeframe_duration_in_seconds * 1000
#     timedelta = limit * timeframe_duration_in_ms
#     all_ohlcv = []
#     while True:
#         fetch_since = earliest_timestamp - timedelta
#         # dt.datetime.fromtimestamp(fetch_since / 1e3) to see fetch_since in date format
#         ohlcv = retry_fetch_ohlcv(exchange, max_retries, symbol, timeframe, fetch_since, limit)
#         if len(ohlcv) > 0:
#             # if we have reached the beginning of history
#             if ohlcv[0][0] >= earliest_timestamp:
#                 break
#             earliest_timestamp = ohlcv[0][0]
#             all_ohlcv = ohlcv + all_ohlcv
#             print(len(all_ohlcv), symbol, 'candles in total from', exchange.iso8601(all_ohlcv[0][0]), 'to',
#                   exchange.iso8601(all_ohlcv[-1][0]))
#             # if we have reached the checkpoint
#             if fetch_since < since:
#                 break
#         else:
#             print(symbol + ' has no market data ')
#             return all_ohlcv
#     return all_ohlcv


def write_to_csv(filename, exchange, data, symbol, append):
    p = Path("./data/raw/", str(exchange))
    p.mkdir(parents=True, exist_ok=True)
    full_path = p / str(filename)

    with Path(full_path).open('w+', newline='') as output_file:
        csv_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerows(data)

    return full_path


def scrape_candles_to_csv(filename, exchange, max_retries, symbol, timeframe, since, limit, append):
    # convert since from string to milliseconds integer if needed
    if isinstance(since, str):
        since = exchange.parse8601(since)

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
        print("No data found for " + symbol + " at ")
        print(exchange)


def import_csv_to_quest_db(file_path, append):
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


def get_candles(file_name, exchange, max_retries, symbol, candle_size, since, limit, append):
    return scrape_candles_to_csv(file_name,
                                 exchange,
                                 max_retries,
                                 symbol,
                                 candle_size,
                                 since,
                                 limit,
                                 append)

# def scrape_ohlcv(exchange, max_retries, symbol, timeframe, since, limit):
#     earliest_timestamp = exchange.milliseconds()
#     timeframe_duration_in_seconds = exchange.parse_timeframe(timeframe)
#     timeframe_duration_in_ms = timeframe_duration_in_seconds * 1000
#
#     all_ohlcv = []
#     while True:
#         if limit > 1000:
#             new_limit = 1000
#             timedelta = new_limit * timeframe_duration_in_ms
#             limit = limit - new_limit
#             fetch_since = earliest_timestamp - timedelta
#             print("========featch since if================")
#             print(dt.datetime.fromtimestamp(fetch_since / 1e3))
#         else:
#             timedelta = limit * timeframe_duration_in_ms
#             fetch_since = earliest_timestamp - timedelta
#             print("========featch since else================")
#             print(dt.datetime.fromtimestamp(fetch_since / 1e3))
#
#         # dt.datetime.fromtimestamp(fetch_since / 1e3) to see fetch_since in date format
#         ohlcv = retry_fetch_ohlcv(exchange, max_retries, symbol, timeframe, fetch_since, limit)
#         if len(ohlcv) > 0:
#             # if we have reached the beginning of history
#             print('*******************************************')
#             if ohlcv[0][0] >= earliest_timestamp:
#                 print(dt.datetime.fromtimestamp(ohlcv[0][0] / 1e3))
#                 print(dt.datetime.fromtimestamp(earliest_timestamp / 1e3))
#                 break
#             earliest_timestamp = ohlcv[0][0]
#             #earliest_timestamp = fetch_since
#             all_ohlcv = ohlcv + all_ohlcv
#             print(len(all_ohlcv), symbol, 'candles in total from', exchange.iso8601(all_ohlcv[0][0]), 'to',
#                   exchange.iso8601(all_ohlcv[-1][0]))
#
#             print("=========bef ex condition=====fetch_since and since==========")
#             print(dt.datetime.fromtimestamp(fetch_since / 1e3))
#             print(dt.datetime.fromtimestamp(since / 1e3))
#             # if we have reached the checkpoint
#             if fetch_since < since:
#                 print("=========exit condition=====fetch_since >= since==========")
#                 print(dt.datetime.fromtimestamp(fetch_since / 1e3))
#                 print(dt.datetime.fromtimestamp(since / 1e3))
#                 break
#         else:
#             print(symbol + ' has no market data ')
#             return all_ohlcv
#     return all_ohlcv
