import utility.constants as const
import utility.util as util
import pandas as pd
from datetime import datetime as dt, timezone as tz, timedelta as td


def load_data():
    try:
        json = util.run_query(const.host, "SELECT * from " + "'" + const.file_name + "'")
        dff = util.format_candle_data(pd.DataFrame(json.get("dataset")))
        dff.drop_duplicates(inplace=True)
        dff = dff.sort_index(axis=0, ascending=True, na_position='last')
        return dff
    except Exception:
        raise "Extract data first"


def since_from_duration(duration, unit):
    now = dt.utcnow()
    print(now)
    return now - td(hours=duration)  # respond with datetime.datetime


def filter_df_by_interval(data_frame, from_date, to_date):
    """Accept the data frame and the date intervals in string format"""
    data_f = data_frame.copy()

    from_date = dt.strptime(from_date, '%Y-%m-%d %H:%M:%S')
    to_date = dt.strptime(to_date, '%Y-%m-%d %H:%M:%S')
    date_index = pd.to_datetime(data_f.index).tz_localize(None)

    return data_f[(date_index >= from_date) & (date_index <= to_date)]


if __name__ == '__main__':
    df = load_data()
    print(df)
