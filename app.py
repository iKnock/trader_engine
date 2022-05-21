import DataminingEngine.read_candles as rd
import utility.constants as const
import utility.util as util


def read_latest_candles():
    try:
        from_date = util.read_csv_last_date()
        if not (from_date and from_date.strip()) or len(from_date) == 0:
            since = const.since
            append = True
        else:
            since = from_date
            append = False

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
                   util.calc_limit(since),
                   append)


if __name__ == '__main__':
    read_latest_candles()
