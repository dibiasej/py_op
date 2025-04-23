from .load_option_chain import TODAY
from .option_data_structures import OptionNode, OptionGraph, EuropeanOptionNode, AmericanOptionNode
from . import processing_utilities as pu

import numpy as np
import datetime as dt

class CreateOption:

    """
    This class is used to create objects of a type of OptionNode we pass in, ex EuropeanOptionNode
    """

    def __init__(self, ticker: str, close_date: str, expiration: str, strike: int) -> None:
        self._ticker: str = ticker
        self._close_date: str = close_date
        self._expiration: str = expiration
        self._strike: int = strike

    def create_call(self, option_node: OptionNode) -> OptionNode:

        chain = pu.cache_chain(self._ticker, self._close_date)
        call_chain = chain[0]
        chain_at_exp = call_chain[self._expiration]

        index: int = np.abs(chain_at_exp['Strike'] - self._strike).argmin()

        exp_date = dt.datetime.strptime(self._expiration, "%Y-%m-%d")
        close_date = dt.datetime.strptime(self._close_date, "%Y-%m-%d")
        dte: int = (exp_date - close_date).days

        strike: int = chain_at_exp['Strike'][index]
        price: float = chain_at_exp['Price'][index]
        bid: float = chain_at_exp['Bid'][index]
        ask: float = chain_at_exp['Ask'][index]
        iv: float = chain_at_exp['Market Implied Volatility'][index]
        vol: int = chain_at_exp['Volume'][index]
        oi: int = chain_at_exp['Open Interest'][index]

        option = option_node(strike, price, bid, ask, iv, self._expiration, dte, vol, oi, 'call')
        return option

    def create_put(self, option_node: OptionNode) -> OptionNode:

        chain = pu.cache_chain(self._ticker, self._close_date)
        put_chain = chain[1]
        chain_at_exp = put_chain[self._expiration]

        index: int = np.abs(chain_at_exp['Strike'] - self._strike).argmin()

        exp_date = dt.datetime.strptime(self._expiration, "%Y-%m-%d")
        close_date = dt.datetime.strptime(self._close_date, "%Y-%m-%d")
        dte: int = (exp_date - close_date).days

        strike: int = chain_at_exp['Strike'][index]
        price: float = chain_at_exp['Price'][index]
        bid: float = chain_at_exp['Bid'][index]
        ask: float = chain_at_exp['Ask'][index]
        iv: float = chain_at_exp['Market Implied Volatility'][index]
        vol: int = chain_at_exp['Volume'][index]
        oi: int = chain_at_exp['Open Interest'][index]

        option = option_node(strike, price, bid, ask, iv, self._expiration, dte, vol, oi, 'put')

        return option
    
class CreateOptionGraph:

    def __init__(self, ticker: str, close_date: str) -> None:
        self._ticker: str = ticker
        self._close_date: str = close_date

    def create_call_graph(self, option_node: OptionNode) -> OptionGraph:

        expirations = pu.call_chain_exp(self._ticker, self._close_date)

        option_graph: OptionGraph = OptionGraph()

        option_graph.close_date = self._close_date

        option_graph.ticker = self._ticker

        for expiration in expirations:

            strikes = pu.call_exp_strikes(self._ticker, self._close_date, expiration)

            for strike in strikes:

                option = CreateOption(self._ticker, self._close_date, expiration, strike).create_call(option_node)

                option_graph.add_option(option)

        return option_graph
    
    def create_put_graph(self, option_node: OptionNode) -> OptionGraph:

        expirations = pu.put_chain_exp(self._ticker, self._close_date)

        option_graph: OptionGraph = OptionGraph()

        option_graph.close_date = self._close_date

        option_graph.ticker = self._ticker

        for expiration in expirations:

            strikes = pu.put_exp_strikes(self._ticker, self._close_date, expiration)

            for strike in strikes:

                option = CreateOption(self._ticker, self._close_date, expiration, strike).create_put(option_node)

                option_graph.add_option(option)

        return option_graph
    
class OptionFactory:
    """
    Date formalt is YYYY-MM-DD
    """

    option_nodes: dict = {
        "european" : EuropeanOptionNode,
        "american" : AmericanOptionNode
    }

    @staticmethod
    def create_option(ticker: str, strike: int, expiration: str, close_date: str = TODAY, option_style: str = "european", option_type: str = "call") -> OptionNode:

        option = OptionFactory.option_nodes.get(option_style.lower())

        if not option:

            return ValueError(f"Unsupported option type {option_style}")
        
        if option_type.lower() == "call":

            return CreateOption(ticker, close_date, expiration, strike).create_call(option)
        
        if option_type.lower() == "put":

            return CreateOption(ticker, close_date, expiration, strike).create_put(option)
        
        else:

            return ValueError(f"Option type {option_type} is not defined")

    @staticmethod 
    def create_option_graph(ticker: str, close_date: str = TODAY, option_style: str = "european", option_type: str = "call"):
        
        option = OptionFactory.option_nodes.get(option_style.lower())

        if not option:

            return ValueError(f"Unsupported option type {option_style}")
        
        if option_type.lower() == "call":

            return CreateOptionGraph(ticker, close_date).create_call_graph(option)
        
        if option_type.lower() == "put":

            return CreateOptionGraph(ticker, close_date).create_put_graph(option)
        
        else:

            return ValueError(f"Option type {option_type} is not defined")

"""Notes: The process call data and put data functions are basically like a surface
            - There might be a lot of overhead because of our calls to functions from processing_utility module, this is a code smell that probably can be cleaned up to speed up the code"""

def main() -> None:

    return None

if __name__ == "__main__":
    print(main())