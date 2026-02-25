from abc import ABC, abstractmethod

"""
I think we can make a position snapshot and a position time series similar to what we did for option chain.
We also might want to recreate a data pipe line specifically for this
    position/backtest repository -> position builder -> position data structure

An issue with this is we need to be able to specify a strike and expiration so we need a way to fetch the data first so this might not work.
It might be faster to get a single option chain snapshot at the starting trading date, then using the data from that option chain to get a new query for the exact option position I want THIS IS A BIG MAYBE!!
In option_chain_builder we 1. get the data (using repositiory) 2. build the data structure -- we can do something along these lines for back testing
"""

class Position(ABC):
    pass

class OptionPosition(Position):

    def __init__(self, price: float, S: float, K: float, T: float, sigma: float = None, r: float = .04, otype: str = "call", q: int = 0, quantity: float = 1) -> None:
        self.price: float = price
        self.S: float = S
        self.K: float = K
        self.T: float = T
        self.sigma: float = sigma
        self.r: float = r
        self.otype: str = otype
        self.q: float = q
        self.quantity: int = quantity

    def value(self) -> float:

        if self.price is None:
            raise ValueError("Market Price is not defined")
        
        else:
            return self.price * self.quantity

    def notional_value(self) -> float:
        
        if self.price is None:
            raise ValueError("Market Price is not defined")
        
        else:
            return self.price * self.quantity * 100
        
    def __repr__(self) -> str:
        return f"Option(price: {self.price}, S: {self.S}, K: {self.K}, dte: {self.T}, type: {self.otype})"
    
class StockPosition(Position):
    """
    For short positions the quantity must be negative
    """

    def __init__(self, ticker: str, price: float, quantity: int = 1) -> None:
        self.ticker: str = ticker
        self.price: float = price
        self.quantity: int = quantity

    def notional_value(self) -> float:
        return self.price * self.quantity
    
    def __repr__(self) -> str:
        return "Stock"
    
class Portfolio:

    def __init__(self) -> None:
        self.positions: list = []

    def add_option_position(self, price: float, S: float, K: float, T: float, sigma: float, r: float, otype: str, q: int = 0, quantity: float = 1) -> None:
        option_position = OptionPosition(price, S, K, T, sigma, r, otype, q, quantity)
        self.positions.append(option_position)

    def add_stock_position(self, price: float, quantity: int) -> None:
        stock_position = StockPosition(price, quantity)
        self.positions.append(stock_position)
    
    def add_position(self, position: StockPosition | OptionPosition) -> None:
        self.positions.append(position)
    
    def notinal_value(self) -> float:
        n_value = 0 
        for position in self.positions:
            p_val = position.notional_value()
            n_value += p_val
        
        return n_value
    
    def __iter__(self):
        for position in self.positions:
            yield position