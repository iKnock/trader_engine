import AnalyticsEngine.transform_data as td
import DataminingEngine.load_data as ld


def run():
    df = ld.load_data()
    df_with_indicators = td.cal_indicators(df)
    return df_with_indicators


if __name__ == '__main__':
    print(run())
