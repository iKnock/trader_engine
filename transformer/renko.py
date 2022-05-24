import numpy as np
from stocktrends import Renko
import statsmodels.api as sm
import transformer.indicators as indicator


def renko_data(data_f, brick_siz=4):
    "function to convert ohlc data into renko bricks"
    df = data_f.copy()
    df.reset_index(inplace=True)
    df.drop("VOLUME", axis=1, inplace=True)  # Axis=1 signify Close is a column and inplace=True in this var not a copy
    df.columns = ["date", "open", "high", "low", "close"]
    df2 = Renko(df)
    df2.brick_size = brick_siz
    renko_df = df2.get_ohlc_data()  # if using older version of the library please use get_bricks() instead
    return renko_df


def renko_dfr(data_f):
    "function to convert ohlc data into renko bricks"
    df = data_f.copy()
    df.reset_index(inplace=True)
    df = df.iloc[:, [0, 1, 2, 3, 4, 5]]
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df2 = Renko(df)
    df2.brick_size = max(0.5, round(indicator.atr(data_f, 120)[-1], 0))
    renko_df = df2.get_ohlc_data()
    renko_df["bar_num"] = np.where(renko_df["uptrend"] == True, 1, np.where(renko_df["uptrend"] == False, -1, 0))
    for i in range(1, len(renko_df["bar_num"])):
        if renko_df.loc[:, "bar_num"][i] > 0 and renko_df.loc[:, "bar_num"][i - 1] > 0:
            renko_df.loc[:, "bar_num"][i] += renko_df.loc[:, "bar_num"][i - 1]
        elif renko_df.loc[:, "bar_num"][i] < 0 and renko_df.loc[:, "bar_num"][i - 1] < 0:
            renko_df.loc[:, "bar_num"][i] += renko_df.loc[:, "bar_num"][i - 1]
    renko_df.drop_duplicates(subset="date", keep="last", inplace=True)
    return renko_df


def slope(ser, n):
    "function to calculate the slope of n consecutive points on a plot"
    slopes = [i * 0 for i in range(n - 1)]
    for i in range(n, len(ser) + 1):
        y = ser[i - n:i]
        x = np.array(range(n))
        y_scaled = (y - y.min()) / (y.max() - y.min())
        x_scaled = (x - x.min()) / (x.max() - x.min())
        x_scaled = sm.add_constant(x_scaled)
        model = sm.OLS(y_scaled, x_scaled)
        results = model.fit()
        slopes.append(results.params[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    return np.array(slope_angle)
