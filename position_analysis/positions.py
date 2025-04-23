from abc import ABC, abstractmethod

class Position(ABC):
    pass

class OptionPosition(Position):

    def __init__(self, price: float, S: float, K: float, T: float, sigma: float, r: float, otype: str, q: int = None, quantity: float = 1) -> None:
        self.price: float = price
        self.S: float = S
        self.K: float = K
        self.T: float = T
        self.sigma: float = sigma
        self.r: float = r
        self.otype: str = otype
        self.q: float = q
        self.quantity: int = quantity

    def position_value(self) -> float:

        if self.price is None:
            raise ValueError("Market Price is not defined")
        
        else:
            return self.price * self.quantity

    def position_notional_value(self) -> float:
        
        if self.price is None:
            raise ValueError("Market Price is not defined")
        
        else:
            return self.price * self.quantity * 100
        
    def __repr__(self) -> str:
        return "Option"
    
class StockPosition(Position):
    """
    For short positions the quantity must be negative
    """

    def __init__(self, price: float, quantity: int = 1) -> None:
        self.price: float = price
        self.quantity: int = quantity

    def position_notional_value(self) -> float:
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
        return self.positions.append(stock_position)
    
    def notinal_value(self) -> float:
        n_value = 0 
        for position in self.positions:
            p_val = position.position_notional_value()
            n_value += p_val
        
        return n_value
    
    def __iter__(self):
        for position in self.positions:
            yield position