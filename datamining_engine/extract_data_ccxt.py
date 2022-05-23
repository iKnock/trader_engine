import datamining_engine.read_candles_ccxt as rd
import utility.constants as const
import utility.util as util
import config.ccxt_wrapper as ccxt_wrapper
import pandas as pd


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


def get_order_book(excnge, ticker):
    orderbook_binance_btc_usdt = excnge.fetch_order_book(ticker)

    bids_binance = orderbook_binance_btc_usdt['bids']
    asks_binanace = orderbook_binance_btc_usdt['asks']

    df_bid_binance = pd.DataFrame(bids_binance, columns=['price', 'qty'])
    df_ask_binance = pd.DataFrame(asks_binanace, columns=['price', 'qty'])

    ticker_order_book = {'bid_binance': df_bid_binance, 'ask_binance': df_ask_binance}
    return ticker_order_book


if __name__ == '__main__':
    read_latest_candles()
