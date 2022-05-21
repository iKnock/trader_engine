import utility.constants as const
import utility.util as util
import pandas as pd
from datetime import datetime as dt, timezone as tz


def load_data():
    host = 'http://localhost:9000'
    json = util.run_query(host, "SELECT * from " + "'" + const.file_name + "'")
    dff = util.format_candle_data(pd.DataFrame(json.get("dataset")))

    dff = dff.sort_index(axis=0, ascending=True, na_position='last')
    dff.drop_duplicates(inplace=True)
    return dff


if __name__ == '__main__':
    df = load_data()
    print(df)
