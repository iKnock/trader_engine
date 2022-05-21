import DataminingEngine.read_candles as rd
import utility.constants as const
import utility.util as util
import pandas as pd
from datetime import datetime as dt, timezone as tz


def read_latest_candles():
    try:
        from_date = util.read_csv_last_date()
        if not (from_date and from_date.strip()) or len(from_date) == 0:
            since = const.since
            append = False
        else:
            since = from_date
            append = False

        fetch_candles(since, append)
    except ValueError:
        append = True
        fetch_candles(const.since, append)


def fetch_candles(since, append):
    rd.get_candles(const.file_name,
                   const.exchange,
                   const.max_retries,
                   const.symbol,
                   const.candle_size,
                   since,
                   util.calc_limit(since),
                   append)


def read_db(today):
    host = 'http://localhost:9000'
    json = util.run_query(host, "SELECT * from " + "'" + const.file_name + "'")
    dff = util.format_candle_data(pd.DataFrame(json.get("dataset")))

    #now = pd.to_datetime(today, utc=True).tz_convert('europe/rome')

    #dff = dff[(dff.index >= const.since) & (dff.index <= now)]
    #dff = dff.sort_index(axis=0, ascending=True, na_position='last')
    dff.drop_duplicates(inplace=True)
    return dff


if __name__ == '__main__':
    df = read_db(dt.utcnow())
    df[df.duplicated()]
    read_latest_candles()
    # util.current_timestamp()
