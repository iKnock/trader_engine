import ccxt


def init_exchange_market(exchange_id):
    # instantiate the exchange by id
    exchange = getattr(ccxt, exchange_id)({
        'enableRateLimit': True,  # required by the Manual
    })
    # preload all markets from the exchange
    exchange.load_markets()
    return exchange
