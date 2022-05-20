import DataminingEngine.read_candles as rd
import utility.constants as const
import utility.util as util
import pandas as pd


def read_latest_candles():
    try:
        from_date = util.read_csv_last_date()
        if not (from_date and from_date.strip()) or len(from_date) == 0:
            since = const.since
            append = False
        else:
            since = from_date
            append = True

        fetch_candles(since, append)
    except ValueError:
        append = True
        fetch_candles(const.since, append)


def fetch_candles(since, append):
    rd.get_candles(const.file_name,
                   const.exchange,
                   const.max_retries,
                   const.symbol,
                   const.candle_size,
                   since,
                   const.limit,
                   append)


def read_db():
    host = 'http://localhost:9000'
    json = util.run_query(host, "SELECT * from 'BTC_euro_30m.csv'")
    df = pd.DataFrame(json.get("dataset"))
    df = util.format_candle_data(df)
    return df

#df=read_db()
#duplicate = df[df.duplicated()]

#pd.DataFrame(df).duplicated().values

#df=pd.DataFrame(util.read_csv_df("./data/raw/Binance/BTC_euro_30m.csv"))
#df = util.format_candle_data(df)

read_latest_candles()

"""
# Creating a DataFrame object
df = pd.DataFrame(employees,
                  columns = ['Name', 'Age', 'City'])
 
# Selecting duplicate rows based
# on 'City' column
duplicate = df[df.duplicated('City')]
 
print("Duplicate Rows based on City :")
 
# Print the resultant Dataframe
duplicate
"""