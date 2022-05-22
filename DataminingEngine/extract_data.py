import DataminingEngine.read_candles as rd
import utility.constants as const
import utility.util as util
import DataminingEngine.ccxt_wrapper as ccxt_wrapper


def read_latest_candles():
    try:
        exchange = ccxt_wrapper.init_exchange_market(const.exchange)
        from_date = util.read_csv_last_date(exchange, const.file_name)
        if not (from_date and from_date.strip()) or len(from_date) == 0:
            since = const.since
            append = True
        else:
            since = from_date
            append = False

        fetch_candles(exchange, since, append)
    except ValueError:
        append = True
        exchange = ccxt_wrapper.init_exchange_market(const.exchange)
        fetch_candles(exchange, const.since, append)


def fetch_candles(exchange, since, append):
    rd.get_candles(const.file_name,
                   exchange,
                   const.max_retries,
                   const.symbol,
                   const.candle_size,
                   since,
                   util.calc_limit(since),
                   append)


if __name__ == '__main__':
    read_latest_candles()
