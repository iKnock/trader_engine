import transformer.transform_data as trans_data
import extract_and_load.load_data as ld
import utility.constants as const
import utility.util as util
from datetime import datetime as dt, timezone as tz, timedelta as td
import utility.visualization_util as vis
import numpy as np
import pandas as pd


def run():
    df = ld.load_data()

    now_str = dt.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    df_filtered = ld.filter_df_by_interval(df, const.since, now_str)
    df_zoned = util.convert_df_timezone(df_filtered)

    df_with_indicators = trans_data.cal_indicators(df_zoned)
    return df_with_indicators


def breakout(data_f):
    ohlc_dict = data_f.copy()

    tickers_ret = [0]
    tickers_signal = ""

    ohlc_dict.dropna(inplace=True)
    df_signal = [0]

    print("calculating returns******************* ")
    for i in range(1, len(ohlc_dict)):
        if tickers_signal == "":
            tickers_ret.append(0)
            df_signal.append(0)
            if ohlc_dict["HIGH"][i] >= ohlc_dict["roll_max_cp"][i] and \
                    ohlc_dict["VOLUME"][i] > 1.5 * ohlc_dict["roll_max_vol"][i - 1]:
                tickers_signal = "Buy"
            elif ohlc_dict["LOW"][i] <= ohlc_dict["roll_min_cp"][i] and \
                    ohlc_dict["VOLUME"][i] > 1.5 * ohlc_dict["roll_max_vol"][i - 1]:
                tickers_signal = "Sell"

        elif tickers_signal == "Buy":
            if ohlc_dict["LOW"][i] < ohlc_dict["CLOSE"][i - 1] - ohlc_dict["ATR"][i - 1]:
                tickers_signal = ""
                df_signal.append("N/A")
                tickers_ret.append(
                    ((ohlc_dict["CLOSE"][i - 1] - ohlc_dict["ATR"][i - 1]) / ohlc_dict["CLOSE"][i - 1]) - 1)
            elif ohlc_dict["LOW"][i] <= ohlc_dict["roll_min_cp"][i] and \
                    ohlc_dict["VOLUME"][i] > 1.5 * ohlc_dict["roll_max_vol"][i - 1]:
                tickers_signal = "Sell"
                df_signal.append(tickers_signal)
                tickers_ret.append((ohlc_dict["CLOSE"][i] / ohlc_dict["CLOSE"][i - 1]) - 1)
            else:
                tickers_ret.append((ohlc_dict["CLOSE"][i] / ohlc_dict["CLOSE"][i - 1]) - 1)
                df_signal.append("N/A")

        elif tickers_signal == "Sell":
            if ohlc_dict["HIGH"][i] > ohlc_dict["CLOSE"][i - 1] + ohlc_dict["ATR"][i - 1]:
                tickers_signal = ""
                df_signal.append(tickers_signal)
                tickers_ret.append(
                    (ohlc_dict["CLOSE"][i - 1] / (ohlc_dict["CLOSE"][i - 1] + ohlc_dict["ATR"][i - 1])) - 1)
            elif ohlc_dict["HIGH"][i] >= ohlc_dict["roll_max_cp"][i] and \
                    ohlc_dict["VOLUME"][i] > 1.5 * ohlc_dict["roll_max_vol"][i - 1]:
                tickers_signal = "Buy"
                df_signal.append(tickers_signal)
                tickers_ret.append((ohlc_dict["CLOSE"][i - 1] / ohlc_dict["CLOSE"][i]) - 1)
            else:
                tickers_ret.append((ohlc_dict["CLOSE"][i - 1] / ohlc_dict["CLOSE"][i]) - 1)
                df_signal.append("N/A")

    ohlc_dict["signal"] = np.array(df_signal)
    ohlc_dict["ret"] = np.array(tickers_ret)
   # print(ohlc_dict)
    return ohlc_dict


if __name__ == '__main__':
    df_ind = run()
    df = df_ind.iloc[:, [0, 1, 2, 3, 4, 7, 14, 15, 16]]
    break_df = breakout(df)
    # to_plot = [renko_data['high'], renko_data['low'], renko_data['close'], renko_data['uptrend']]
    # vis.plot_data_many(to_plot, 'BTC/EUR Closing price 24 hours, 5Min',
    #                   'Time', 'Price', 'Bou Band')
    print(break_df)
