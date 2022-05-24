import numpy as np


def breakout(data_f):
    ohlc_dict = data_f.copy()
    ohlc_dict = ohlc_dict.iloc[:, [0, 1, 2, 3, 4, 7, 14, 15, 16]]
    ohlc_dict.dropna(inplace=True)

    tickers_ret = [0]
    tickers_signal = ""
    df_signal = [0]

    print("calculating returns******************* ")
    for i in range(1, len(ohlc_dict)):
        if tickers_signal == "":
            tickers_ret.append(0)

            if ohlc_dict["HIGH"][i] >= ohlc_dict["roll_max_cp"][i] and \
                    ohlc_dict["VOLUME"][i] > 1.5 * ohlc_dict["roll_max_vol"][i - 1]:
                tickers_signal = "Buy"
            elif ohlc_dict["LOW"][i] <= ohlc_dict["roll_min_cp"][i] and \
                    ohlc_dict["VOLUME"][i] > 1.5 * ohlc_dict["roll_max_vol"][i - 1]:
                tickers_signal = "Sell"

            if tickers_signal == 'Sell' or tickers_signal == 'Buy':
                df_signal.append(tickers_signal)
            else:
                df_signal.append("N/A")

        elif tickers_signal == "Buy":
            if ohlc_dict["LOW"][i] < ohlc_dict["CLOSE"][i - 1] - ohlc_dict["ATR"][i - 1]:
                tickers_signal = ""
                df_signal.append(
                    "Sell/Stop Loss")  # selling price will be ohlc_dict["CLOSE"][i - 1] - ohlc_dict["ATR"][i - 1]
                tickers_ret.append(
                    ((ohlc_dict["CLOSE"][i - 1] - ohlc_dict["ATR"][i - 1]) / ohlc_dict["CLOSE"][i - 1]) - 1)
            elif ohlc_dict["LOW"][i] <= ohlc_dict["roll_min_cp"][i] and \
                    ohlc_dict["VOLUME"][i] > 1.5 * ohlc_dict["roll_max_vol"][i - 1]:
                tickers_signal = "Sell"  # trend reversal
                df_signal.append(tickers_signal)
                tickers_ret.append((ohlc_dict["CLOSE"][i] / ohlc_dict["CLOSE"][i - 1]) - 1)
            else:
                tickers_ret.append((ohlc_dict["CLOSE"][i] / ohlc_dict["CLOSE"][i - 1]) - 1)
                df_signal.append("Buy")

        elif tickers_signal == "Sell":
            if ohlc_dict["HIGH"][i] > ohlc_dict["CLOSE"][i - 1] + ohlc_dict["ATR"][i - 1]:
                tickers_signal = ""
                df_signal.append("stop loss/stop shorting")
                tickers_ret.append(
                    (ohlc_dict["CLOSE"][i - 1] / (ohlc_dict["CLOSE"][i - 1] + ohlc_dict["ATR"][i - 1])) - 1)
            elif ohlc_dict["HIGH"][i] >= ohlc_dict["roll_max_cp"][i] and \
                    ohlc_dict["VOLUME"][i] > 1.5 * ohlc_dict["roll_max_vol"][i - 1]:
                tickers_signal = "Buy"  # trend reversal
                df_signal.append(tickers_signal)
                tickers_ret.append((ohlc_dict["CLOSE"][i - 1] / ohlc_dict["CLOSE"][i]) - 1)
            else:
                tickers_ret.append((ohlc_dict["CLOSE"][i - 1] / ohlc_dict["CLOSE"][i]) - 1)
                df_signal.append("Sell/Continue Shorting)")

    ohlc_dict["signal"] = np.array(df_signal)
    ohlc_dict["ret"] = np.array(tickers_ret)
    return ohlc_dict


if __name__ == '__main__':
    br_out = breakout()
    print(br_out)
