import datetime as dt
import yfinance as yf
import pandas as pd

from data.price_data import process_price_data

MARKET_HOLIDAYS: list[str] = ["2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19", "2024-07-04", "2024-08-02", "2024-11-28", "2024-12-25"]
MISSED_DAYS: list[str] = ["2024-01-04"]

TODAY: str = dt.datetime.now().strftime('%Y-%m-%d')

def get_stock_price(ticker, date):
    return process_price_data.get_close_prices(ticker, date)[0][0]

def get_atm_option(o_graph, exp, date = TODAY, stock_price: float = None):
    """
    Used on OptionGraph() in option_data_structures to get the atm option for a given expiration
    """

    skew = o_graph.get_skew(exp)
    ticker = o_graph.ticker
    
    if stock_price is None:
        current_price = get_stock_price(ticker, date)

    else:
        
        current_price = stock_price

    first = 0
    last = len(skew.strikes()) - 1

    smallest_diff = float("inf")
    best_strike = None

    while first <= last:
        mid = (first + last)//2

        diff = abs(skew[mid].get_strike() - current_price)

        if diff <= smallest_diff:
            smallest_diff = diff
            best_strike = skew[mid]

        if diff < 1:
            return skew[mid]
        
        else:
            if skew[mid].get_strike() < current_price:
                first = mid + 1

            else:
                last = mid - 1

    return best_strike