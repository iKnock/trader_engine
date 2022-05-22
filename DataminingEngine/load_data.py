import utility.constants as const
import utility.util as util
import pandas as pd
from datetime import datetime as dt, timezone as tz, timedelta as td


def load_data():
    host = 'http://localhost:9000'
    json = util.run_query(host, "SELECT * from " + "'" + const.file_name + "'")
    dff = util.format_candle_data(pd.DataFrame(json.get("dataset")))
    #dff = pd.DataFrame(json.get("dataset"))
    # now = pd.to_datetime(today, utc=True).tz_convert('europe/rome')
    now = dt.utcnow()

  #  now = pd.to_datetime(now, utc=True).tz_convert('europe/rome')
 #   since = str(now - filter_by_duration(24, "hours"))

    since = pd.to_datetime(const.since, utc=True).tz_convert('europe/rome')

#    dff = dff[(dff.index >= since) & (dff.index <= now)]

    dff.drop_duplicates(inplace=True)
    dff = dff.sort_index(axis=0, ascending=True, na_position='last')
    return dff


def filter_by_duration(duration, unit):
    return td(hours=duration)


if __name__ == '__main__':
    df = load_data()
    print(df)
