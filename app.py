import transformer.transform_data as trans_data
import extract_and_load.load_data as ld
import utility.constants as const
import utility.util as util
from datetime import datetime as dt, timezone as tz, timedelta as td
import utility.visualization_util as vis
import strategy.breackout as br_out
import strategy.renko_obv as renko_obv
import strategy.renko_macd as renko_macd
import transformer.indicators as indicator
import numpy as np
import pandas as pd
import performance_evaluation.performace_kpi as kpi
import strategy.signal as sgl


def measure_performance(d_frame):
    # calculating overall strategy's KPIs
    data_f = d_frame.copy()
    data_f["ret_mean"] = data_f["ret"].mean()

    performance = pd.DataFrame()
    performance["close"] = data_f["CLOSE"]

    performance["cagr"] = kpi.CAGR(data_f, "ret")
    performance["sharpe"] = kpi.sharpe(data_f, 0.025)
    performance["ret"] = data_f["ret"]
    performance["ret_mean"] = kpi.max_dd(data_f, "ret_mean")
    return performance


def draw_chart(series_to_plot):
    # visualizing strategy returns
    (1 + series_to_plot).cumprod().plot()


if __name__ == '__main__':
    # load data
    df = ld.load_data()

    # filter data by date
    now_str = dt.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    df_filtered = ld.filter_df_by_interval(df, const.since, now_str)

    # convert df timezone
    #df_filtered = util.convert_df_timezone(df_filtered)

    # add indicators
    df_with_indicators = trans_data.cal_indicators(df_filtered)

    # apply strategy
    #  break_df = br_out.breakout(df_with_indicators)
    # ren_obv = renko_obv.run(df_with_indicators)
    # renk_macd = renko_macd.run(df_with_indicators)

    renko_merge_with_candle = renko_macd.merge_dfs(df_with_indicators)
    signl = sgl.trade_signal(renko_merge_with_candle, "")

    # generate kpi report
    # kpi_report = measure_performance(renk_macd)

    # plot strategy return
    #   draw_chart(break_df['ret'])
    #   draw_chart(ren_obv['ret'])
    #    draw_chart(renk_macd['ret'])

    #   vis.plot_data(break_df['ret'], 'BTC/EUR Closing price 24 hours', '5Min', 'Time', 'Price')

    print(signl)
