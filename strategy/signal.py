import numpy as np
import copy
import strategy.renko_macd as rnk_macd
import extract_and_load.load_data as ld
from datetime import datetime as dt, timezone as tz, timedelta as td
import utility.constants as const
import transformer.transform_data as transf_data


def trade_signal(candle_renko_merged_df, l_s):
    "function to generate signal"
    signal = ""
    df = copy.deepcopy(candle_renko_merged_df)
    df_signal = ['N/A']
    for i in range(len(df)):
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


def trade_signall(candle_renko_merged_df, l_s):
    "function to generate signal"
    tickers_signal = ""
    df = copy.deepcopy(candle_renko_merged_df)
    df_signal = ['N/A']
    for i in range(len(df)):
        if l_s == "":
            if i > 0:
                if df["bar_num"][i] >= 2 and df["macd"][i] > \
                        df["macd_sig"][i] and df["macd_slope"][i] > \
                        df["macd_sig_slope"][i]:
                    tickers_signal = "Buy"
                elif df["bar_num"][i] <= -2 and df["macd"][i] < \
                        df["macd_sig"][i] and df["macd_slope"][i] < \
                        df["macd_sig_slope"][i]:
                    tickers_signal = "Sell"

                if tickers_signal == 'Sell' or tickers_signal == 'Buy':
                    df_signal.append(tickers_signal)
                else:
                    df_signal.append("N/A")

        elif l_s == "long":
            if i > 0:
                if df["bar_num"][i] <= -2 and df["macd"][i] < \
                        df["macd_sig"][i] and df["macd_slope"][i] < \
                        df["macd_sig_slope"][i]:
                    tickers_signal = "Close_Sell"  # Close_sell
                elif df["macd"][i] < df["macd_sig"][i] and \
                        df["macd_slope"][i] < df["macd_sig_slope"][i]:
                    tickers_signal = "Close"

            if tickers_signal == 'Sell' or tickers_signal == 'Buy':
                df_signal.append(tickers_signal)
            else:
                df_signal.append("Close")

        elif l_s == "short":
            if i > 0:
                if df["bar_num"][i] >= 2 and df["macd"][i] > \
                        df["macd_sig"][i] and df["macd_slope"][i] > \
                        df["macd_sig_slope"][i]:
                    tickers_signal = "Close_Buy"
                elif df["macd"][i] > df["macd_sig"][i] and \
                        df["macd_slope"][i] > df["macd_sig_slope"][i]:
                    tickers_signal = "Close"

            if tickers_signal == 'Sell' or tickers_signal == 'Buy':
                df_signal.append(tickers_signal)
            else:
                df_signal.append("Close")

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


if __name__ == '__main__':
    main()
