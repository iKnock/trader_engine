import utility.constants as const
import transformer.transform_data as trans_data
import extract_and_load.load_data as ld
from datetime import datetime as dt, timezone as tz, timedelta as td
import matplotlib.pyplot as plt
import time

upward_sma_dir = {}
dnward_sma_dir = {}


def stochastic(df, a, b, c):
    "function to calculate stochastic"
    df['k'] = ((df['CLOSE'] - df['LOW'].rolling(a).min()) / (
            df['HIGH'].rolling(a).max() - df['LOW'].rolling(a).min())) * 100
    df['K'] = df['k'].rolling(b).mean()
    df['D'] = df['K'].rolling(c).mean()
    return df


def sma(df, a, b):
    "function to calculate stochastic"
    df['sma_fast'] = df['CLOSE'].rolling(a).mean()
    df['sma_slow'] = df['CLOSE'].rolling(b).mean()
    return df


def trade_signal(df):
    "function to generate signal"
    global upward_sma_dir, dnward_sma_dir
    signal = ""
    if df['sma_fast'][-1] > df['sma_slow'][-1] and df['sma_fast'][-2] < df['sma_slow'][-2]:
        upward_sma_dir = True
        dnward_sma_dir = False
    if df['sma_fast'][-1] < df['sma_slow'][-1] and df['sma_fast'][-2] > df['sma_slow'][-2]:
        upward_sma_dir = False
        dnward_sma_dir = True
    if upward_sma_dir == True and min(df['K'][-1], df['D'][-1]) > 25 and max(df['K'][-2], df['D'][-2]) < 25:
        signal = "Buy"
    if dnward_sma_dir == True and min(df['K'][-1], df['D'][-1]) < 75 and max(df['K'][-2], df['D'][-2]) > 75:
        signal = "Sell"

    plt.subplot(211)
    plt.plot(df.iloc[-50:, [3, -2, -1]])
    plt.title('SMA Crossover & Stochastic')
    plt.legend(('close', 'sma_fast', 'sma_slow'), loc='upper left')

    plt.subplot(212)
    plt.plot(df.iloc[-50:, [-4, -3]])
    plt.hlines(y=25, xmin=0, xmax=50, linestyles='dashed')
    plt.hlines(y=75, xmin=0, xmax=50, linestyles='dashed')
    plt.show()

    return signal


def main():
    df = ld.load_data()
    now_str = dt.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    df = ld.filter_df_by_interval(df, const.since, now_str)
    df = trans_data.cal_indicators(df)

    ohlc_df = stochastic(df, 14, 3, 3)
    ohlc_df = sma(ohlc_df, 100, 200)

    signal = trade_signal(ohlc_df)

    if signal == "Buy":
        # market_order(currency, pos_size, 3 * ATR(data, 120))
        print("New long position initiated ")
    elif signal == "Sell":
        # market_order(currency, -1 * pos_size, 3 * ATR(data, 120))
        print("New short position initiated for ")
