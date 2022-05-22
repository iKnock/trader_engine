import pandas as pd
import AnalyticsEngine.indicators as indicators


def cal_indicators(DF):
    df = DF.copy()
    df_with_indicator = pd.DataFrame(indicators.calc_and_add_indicators(df))
    print(df_with_indicator)
    return df_with_indicator
