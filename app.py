import AnalyticsEngine.transform_data as trans_data
import datamining_engine.load_data as ld
import utility.constants as const
import utility.util as util
from datetime import datetime as dt, timezone as tz, timedelta as td
import utility.visualization_util as vis


def run():
    df = ld.load_data()

    now_str = dt.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    df_filtered = ld.filter_df_by_interval(df, const.since, now_str)
    df_zoned = util.convert_df_timezone(df_filtered)

    df_with_indicators = trans_data.cal_indicators(df_zoned)
    return df_with_indicators


if __name__ == '__main__':
    df_ind = run()

    # to_plot = [renko_data['high'], renko_data['low'], renko_data['close'], renko_data['uptrend']]
    # vis.plot_data_many(to_plot, 'BTC/EUR Closing price 24 hours, 5Min',
    #                   'Time', 'Price', 'Bou Band')
    print(df_ind)
