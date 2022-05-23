import numpy as np


def breakout(data_f):
    ohlc_dict = data_f.copy()
    tickers_ret = [0]
    tickers_signal = ""
    ohlc_dict.dropna(inplace=True)

    print("calculating returns******************* ")
    for i in range(1, len(ohlc_dict)):
        if tickers_signal == "":
            tickers_ret.append(0)
            if ohlc_dict["HIGH"][i] >= ohlc_dict["roll_max_cp"][i] and \
                    ohlc_dict["VOLUME"][i] > 1.5 * ohlc_dict["roll_max_vol"][i - 1]:
                tickers_signal = "Buy"
                ohlc_dict['SIGNAL'] = ["Buy"]
            elif ohlc_dict["LOW"][i] <= ohlc_dict["roll_min_cp"][i] and \
                    ohlc_dict["VOLUME"][i] > 1.5 * ohlc_dict["roll_max_vol"][i - 1]:
                tickers_signal = "Sell"
                ohlc_dict['SIGNAL'] = ["Sell"]

        elif tickers_signal == "Buy":
            if ohlc_dict["LOW"][i] < ohlc_dict["CLOSE"][i - 1] - ohlc_dict["ATR"][i - 1]:
                tickers_signal = ""
                ohlc_dict['SIGNAL'] = [""]
                tickers_ret.append(
                    ((ohlc_dict["CLOSE"][i - 1] - ohlc_dict["ATR"][i - 1]) / ohlc_dict["CLOSE"][i - 1]) - 1)
            elif ohlc_dict["LOW"][i] <= ohlc_dict["roll_min_cp"][i] and \
                    ohlc_dict["VOLUME"][i] > 1.5 * ohlc_dict["roll_max_vol"][i - 1]:
                tickers_signal = "Sell"
                ohlc_dict['SIGNAL'] = ["Sell"]
                tickers_ret.append((ohlc_dict["CLOSE"][i] / ohlc_dict["CLOSE"][i - 1]) - 1)
            else:
                tickers_ret.append((ohlc_dict["CLOSE"][i] / ohlc_dict["CLOSE"][i - 1]) - 1)
                ohlc_dict['SIGNAL'] = ["Buy"]

        elif tickers_signal == "Sell":
            if ohlc_dict["HIGH"][i] > ohlc_dict["CLOSE"][i - 1] + ohlc_dict["ATR"][i - 1]:
                tickers_signal = ""
                ohlc_dict['SIGNAL'] = [""]
                tickers_ret.append(
                    (ohlc_dict["CLOSE"][i - 1] / (ohlc_dict["CLOSE"][i - 1] + ohlc_dict["ATR"][i - 1])) - 1)
            elif ohlc_dict["HIGH"][i] >= ohlc_dict["roll_max_cp"][i] and \
                    ohlc_dict["VOLUME"][i] > 1.5 * ohlc_dict["roll_max_vol"][i - 1]:
                tickers_signal = "Buy"
                ohlc_dict['SIGNAL'] = ["Buy"]
                tickers_ret.append((ohlc_dict["CLOSE"][i - 1] / ohlc_dict["CLOSE"][i]) - 1)
            else:
                tickers_ret.append((ohlc_dict["CLOSE"][i - 1] / ohlc_dict["CLOSE"][i]) - 1)
                ohlc_dict['SIGNAL'] = [""]

    ohlc_dict["ret"] = np.array(tickers_ret)
    print(np.array(ohlc_dict["ret"]))
    return ohlc_dict


breakout()