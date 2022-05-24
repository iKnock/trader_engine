import transformer.transform_data as trans_data
import extract_and_load.load_data as ld
import utility.constants as const
import utility.util as util
from datetime import datetime as dt, timezone as tz, timedelta as td
import utility.visualization_util as vis
import strategy.breackout as br_out
import strategy.renko_obv as renko_obv
import transformer.indicators as indicator


def run():
    df = ld.load_data()

    now_str = dt.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    df_filtered = ld.filter_df_by_interval(df, const.since, now_str)
    df_filtered = util.convert_df_timezone(df_filtered)

    df_with_indicators = trans_data.cal_indicators(df_filtered)
    return df_with_indicators


if __name__ == '__main__':
    df_ind = run()

    break_df = br_out.breakout(df_ind)
    ren_obv = renko_obv.run(df_ind.iloc[:, [0, 1, 2, 3, 4]])

    # to_plot = [renko_data['high'], renko_data['low'], renko_data['close'], renko_data['uptrend']]
    # vis.plot_data_many(to_plot, 'BTC/EUR Closing price 24 hours, 5Min',
    #                   'Time', 'Price', 'Bou Band')
    print(ren_obv)
