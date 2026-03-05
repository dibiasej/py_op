from itertools import groupby

from py_op.data.repositories.option_chain_repository import OptionChainRepository
from py_op.data.structures.option_data_structures import OptionChain, OptionContract
from py_op.utils.util_funcs import get_stock_price
from py_op.data.price_data.process_price_data import get_close_prices
from py_op.utils.option_chain_schema import *
#from calc_engine.greeks.analytical_greeks import AnalyticalDelta
#from calc_engine.vol_engine.iv_calc import RootFinder

def create_option_contract(row, S, r, q, steps) -> OptionContract:
    strike = row[STRIKE]
    if strike % steps != 0:
        return None

    close_price = row[CLOSE_PRICE]
    bid = row[BID]
    ask = row[ASK]
    yahoo_iv = row[YAHOO_IV]
    exp = row[EXPIRY]
    dte = row[DTE]
    volume = row[VOLUME]
    open_interest = row[OPEN_INTEREST]
    otype = row[OTYPE]

    # avoid T=0
    if dte == 0:
        dte = 0.5

    T = dte / 252

    #iv = RootFinder().calculate(close_price, S, strike, T, r=r, otype=otype, q=q)
    #delta = AnalyticalDelta().calculate(S, strike, T, iv, r=r, q=q, otype=otype)
    iv, delta = None, None

    return OptionContract(S, strike, close_price, bid, ask, yahoo_iv, exp, dte, volume, open_interest, otype, iv, delta)

def create_chain(ticker: str, close_date: str, r: float = .04, q: float = 0, strike_min: float = None, strike_max: float = None, dte_min: float = None, dte_max: float = None, moneyness: float = None, steps: int = 1):
    """
    Practice function for feeding data from my database into my OptionChain data structure
    Current flow
        DB -> chain repository -> create chain (processor) -> OptionContract (Node) -> OptionChain (Graph) -> Services (Vol, Price, backtest)

    I need to find a way to incorporate parity iv somewhere. I might add a method to OptionChain that calculates parity iv
    """
    S = get_stock_price(ticker, close_date)
    option_chain = OptionChain()
    option_chain.S = S
    option_chain.close_date = close_date

    if moneyness is not None:
        strike_min = S * (1 - moneyness)
        strike_max = S * (1 + moneyness)

    option_chain_data = OptionChainRepository().get_chain_snapshot(ticker, close_date, strike_min=strike_min, strike_max=strike_max, dte_min=dte_min, dte_max=dte_max)

    for row in option_chain_data:
        node = create_option_contract(row, S, r, q, steps)
        if node is not None:
            option_chain.add_option(node)
        
    return option_chain

def create_chain_series(ticker: str, start_date: str, end_date: str, r: float = .04, q: float = 0, strike_min: float = None, strike_max: float = None, dte_min: float = None, dte_max: float = None, steps: int = 1, moneyness: float = None):
    close_price_data = get_close_prices(ticker, start_date, end_date)
    price_date_dict = dict(zip(close_price_data[1], close_price_data[0]))

    option_chain_data_series = OptionChainRepository().get_chain_timeseries(ticker, start_date, end_date, strike_min=strike_min, strike_max=strike_max, dte_min=dte_min, dte_max=dte_max, moneyness=moneyness)
    option_chain_timeseries = []

    for date, option_chain_data in groupby(option_chain_data_series, key=lambda r: r[CLOSE_DATE]):

        S = price_date_dict.get(date)
        if S is None:
            continue
        
        option_chain = OptionChain()
        option_chain.S = S
        option_chain.close_date = date

        for row in option_chain_data:
            node = create_option_contract(row, S, r, q, steps)
            if node is not None:
                option_chain.add_option(node)

        option_chain_timeseries.append(option_chain)

    return option_chain_timeseries