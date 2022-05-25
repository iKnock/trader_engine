import copy
import strategy.renko_macd as rnk_macd
import extract_and_load.load_data as ld
from datetime import datetime as dt, timezone as tz, timedelta as td
import utility.constants as const
import transformer.transform_data as transf_data
import numpy as np


def trade_signal(candle_renko_merged_df, l_s):
    "function to generate signal"
    signal = ""
    df = copy.deepcopy(candle_renko_merged_df)
    df_signal = ['N/A']
    for i in range(1, len(df)):
        if l_s == "":
            if df["bar_num"].tolist()[-1] >= 2 and df["macd"].tolist()[-1] > df["macd_sig"].tolist()[-1] and \
                    df["macd_slope"].tolist()[-1] > df["macd_sig_slope"].tolist()[-1]:
                signal = "Buy"
            elif df["bar_num"].tolist()[-1] <= -2 and df["macd"].tolist()[-1] < df["macd_sig"].tolist()[-1] and \
                    df["macd_slope"].tolist()[-1] < df["macd_sig_slope"].tolist()[-1]:
                signal = "Sell"

            if signal == 'Sell' or signal == 'Buy':
                df_signal.append(signal)
            else:
                df_signal.append("N/A")

        elif l_s == "long":
            if df["bar_num"].tolist()[-1] <= -2 and df["macd"].tolist()[-1] < df["macd_sig"].tolist()[-1] and \
                    df["macd_slope"].tolist()[-1] < df["macd_sig_slope"].tolist()[-1]:
                signal = "Close_Sell"
            elif df["macd"].tolist()[-1] < df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1] < \
                    df["macd_sig_slope"].tolist()[-1]:
                signal = "Close"

            if signal == 'Close_Sell' or signal == 'Close':
                df_signal.append(signal)
            else:
                df_signal.append("N/A")

        elif l_s == "short":
            if df["bar_num"].tolist()[-1] >= 2 and df["macd"].tolist()[-1] > df["macd_sig"].tolist()[-1] and \
                    df["macd_slope"].tolist()[-1] > df["macd_sig_slope"].tolist()[-1]:
                signal = "Close_Buy"
            elif df["macd"].tolist()[-1] > df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1] > \
                    df["macd_sig_slope"].tolist()[-1]:
                signal = "Close"

            if signal == 'Close_Buy' or signal == 'Close':
                df_signal.append(signal)
            else:
                df_signal.append("N/A")

    df["signal"] = np.array(df_signal)
    return df


def main():
    df = ld.load_data()
    now_str = dt.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    df = ld.filter_df_by_interval(df, const.since, now_str)

    # add indicators
    df_with_indicators = transf_data.cal_indicators(df)

    mrg_renko = rnk_macd.merge_dfs(df_with_indicators)
    trade_sign = trade_signal(mrg_renko, "")
    print(trade_sign)
