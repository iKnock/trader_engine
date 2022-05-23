import AnalyticsEngine.indicators as indicators
import datamining_engine.load_data as ld
import utility.constants as const
import utility.util as util
import pandas as pd
from datetime import datetime as dt, timezone as tz, timedelta as td


def cal_indicators(data_f):
    df = data_f.copy()
    df_with_indicator = pd.DataFrame(indicators.calc_and_add_indicators(df))
    print(df_with_indicator)
    return df_with_indicator


def generate_renko(data_f):
    df = data_f.copy()
    return indicators.renko_data(df, "hourlydf")


if __name__ == '__main__':
    ohclv_df = ld.load_data()

    now_str = dt.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    df_filtered = ld.filter_df_by_interval(ohclv_df, const.since, now_str)
    df_filtered = util.convert_df_timezone(df_filtered)

    df_with_indicators = cal_indicators(df_filtered)
    renok_data = generate_renko(df_filtered)

    print(df_with_indicators)
