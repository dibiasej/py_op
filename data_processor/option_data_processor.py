import numpy as np
from itertools import chain

from data.option_data.process_option_chain import OptionFactory
from calc_engine.volatility import iv_calc
from utils.util_funcs import get_stock_price

"""
Need to make a method or function that takes in two array one put one call, they should also be lists with tuples of corresponding strikes.
EX: puts = [(K, model price), (K, model price), (K, model price), (K, model price), (K, model price), (K, model price)]
    calls = [(K, model price), (K, model price), (K, model price), (K, model price), (K, model price), (K, model price)]
    then we make a function given these two lists and a spot price and make a loop using a itertools function to look and compare with a 
    condition saying if K < S use puts else use calls model price
"""

class OptionDataProcessor:

    def __init__(self, ticker: str, close_date: str):
        self.option_call_graph = OptionFactory().create_option_graph(ticker, close_date, option_type = 'call')
        self.option_put_graph = OptionFactory().create_option_graph(ticker, close_date, option_type = 'put')
        self.iv_method = iv_calc.ImpliedVolatility().root_finder
        self.S = get_stock_price(ticker, close_date)

    def set_iv_method(self):
        """
        Create a method that takes a string of root finder, Newton or Bisection and sets that as the iv method.
        """
        pass

    def otm_put_call_prices(self, exp: str, S: float = None) -> list[float]:
        """
        Return a numpy array of otm put and call prices for a given expiration date.
        """
        if S is None:
            S = self.S

        call_skew = self.option_call_graph.get_skew(exp)
        put_skew = self.option_put_graph.get_skew(exp)

        call_strikes = np.array(call_skew.strikes())
        put_strikes = np.array(put_skew.strikes())

        common_strikes = np.intersect1d(call_strikes, put_strikes)

        data = []

        for strike in common_strikes:

            if strike <= S:
                put_price = self.option_put_graph.get_option(exp, strike).get_price()
                data.append((put_price, strike))

            elif strike >= S:
                call_price = self.option_put_graph.get_option(exp, strike).get_price()
                data.append((call_price, strike))

        return data
    
    def otm_put_call_ivs(self, exp: str, S: float = None) -> list[float]:
        """
        Return a numpy array of otm put and call ivs for a given expiration date.
        """
        if S is None:
            S = self.S

        call_skew = self.option_call_graph.get_skew(exp)
        put_skew = self.option_put_graph.get_skew(exp)

        call_strikes = np.array(call_skew.strikes())
        put_strikes = np.array(put_skew.strikes())

        common_strikes = np.intersect1d(call_strikes, put_strikes)
        dte = call_skew.get_dte()
        T = dte / 252
        iv_list = []

        for strike in common_strikes:
            if strike <= S:

                put_price = self.option_put_graph.get_option(exp, strike).get_price()
                op_iv = self.iv_method(put_price, S, strike, T, otype = 'put')
                iv_list.append((op_iv, strike))

            elif strike >= S:

                call_price = self.option_put_graph.get_option(exp, strike).get_price()
                op_iv = self.iv_method(call_price, S, strike, 30/252, otype = 'call')
                iv_list.append((op_iv, strike))

        return iv_list
    
    def otm_put_call_data(self, exp: str, S: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return a tuple of three arrays of otm put and call data including prices, strikes and ivs for a given expiration date.
        """

        if S is None:
            S = self.S

        call_skew = self.option_call_graph.get_skew(exp)
        put_skew = self.option_put_graph.get_skew(exp)

        call_strikes = np.array(call_skew.strikes())
        put_strikes = np.array(put_skew.strikes())

        common_strikes = np.intersect1d(call_strikes, put_strikes)
        dte = call_skew.get_dte()
        T = dte / 252
        result = []

        for strike in common_strikes:
            if strike <= S:
                put_price = self.option_put_graph.get_option(exp, strike).get_price()
                op_iv = self.iv_method(put_price, S, strike, T, otype='put')
                result.append((strike, put_price, op_iv))

            elif strike >= S:
                call_price = self.option_call_graph.get_option(exp, strike).get_price()  # <-- fixed: use call graph
                op_iv = self.iv_method(call_price, S, strike, T, otype='call')
                result.append((strike, call_price, op_iv))

        return result
    
    def select_model_prices(self, puts, calls, S):
        """
        Selects model prices based on strike and spot price.
        
        Parameters:
        - puts: List of tuples [(K, model_price), ...] for put options
        - calls: List of tuples [(K, model_price), ...] for call options
        - S: Spot price

        Returns:
        - List of selected model prices: use put price if K < S, else call price
        """
        selected_prices = []

        # Combine puts and calls using chain for unified iteration
        for (K, price) in chain(puts, calls):
            if K < S:
                # Only include the put price if strike is less than spot
                selected_prices.append(price)
            else:
                # Only include the call price if strike is greater than or equal to spot
                selected_prices.append(price)

        return selected_prices