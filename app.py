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
    json = util.run_query(host, "SELECT * from " + "'" + const.file_name + "'")
    dff = util.format_candle_data(pd.DataFrame(json.get("dataset")))
    dff = dff[(dff.index >= +"'" + const.since + "'") & (dff.index <= '2022-05-20 22:00:00')]
    dff = dff.sort_index(axis=0, ascending=True, na_position='last')
    dff.drop_duplicates(inplace=True)
    return dff


#df = read_db()

# df=pd.DataFrame(util.read_csv_df("./data/raw/Binance/BTC_euro_30m.csv"))
# df = util.format_candle_data(df)

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
