from abc import ABC, abstractmethod

class OptionNode(ABC):

    """class that acts as a data hub for our option and stores all our values corresponding to the specific option"""

    def __init__(self, strike: int = None, price: float = None, bid: float = None, ask: float = None, implied_volatility: float = None,
                expiration: str = None, dte: int = None, volume: int = None, open_interest: int = None, otype: str = None) -> None:

        self._strike: int = strike
        self._price: float = price
        self._bid: float = bid
        self._ask: float = ask
        self._implied_volatility: float = implied_volatility
        self._expiration: str = expiration
        self._dte: int = dte
        self._volume: int = volume
        self._open_interest: int = open_interest
        self.otype: str = otype
    
    def get_strike(self) -> int:
        return self._strike
    
    def get_price(self) -> float:
        return self._price
    
    def set_price(self, value: float) -> None:
        self._price = value

    def get_bid(self) -> float:
        return self._bid

    def get_ask(self) -> float:
        return self._ask
    
    def get_mid_price(self) -> float:
        return (self._bid + self._ask) / 2
    
    def get_implied_volatility(self) -> float:
        return self._implied_volatility
    
    def set_implied_volatility(self, value: float) -> None:
        self._implied_volatility = value

    def get_expiration(self) -> str:
        return self._expiration
    
    def get_dte(self) -> int:
        return self._dte

    def get_volume(self) -> str:
        return self._volume
    
    def get_open_interest(self) -> str:
        return self._open_interest
    
class EuropeanOptionNode(OptionNode):

    """class that acts as a data hub for our option and stores all our values corresponding to the specific option"""

    def __init__(self, strike: int, price: float, bid: float, ask: float, implied_volatility: float, expiration: str, dte: int, volume: int, open_interest: int, otype: str) -> None:

        self._strike: int = strike
        self._price: float = price
        self._bid: float = bid
        self._ask: float = ask
        self._implied_volatility: float = implied_volatility
        self._expiration: str = expiration
        self._dte: int = dte
        self._volume: int = volume
        self._open_interest: int = open_interest
        self.otype: str = otype

    def __repr__(self):
        return (f"EuropeanOptionNode(strike={self._strike}, price={self._price}, expiration={self._expiration})")
    
class AmericanOptionNode(OptionNode):

    """class that acts as a data hub for our option and stores all our values corresponding to the specific option"""

    def __init__(self, strike: int, price: float, bid: float, ask: float, implied_volatility: float, expiration: str, dte: int, volume: int, open_interest: int, otype: str) -> None:

        self._strike: int = strike
        self._price: float = price
        self._bid: float = bid
        self._ask: float = ask
        self._implied_volatility: float = implied_volatility
        self._expiration: str = expiration
        self._dte: int = dte
        self._volume: int = volume
        self._open_interest: int = open_interest
        self.otype: str = otype

    def __repr__(self):
        return (f"AmericanOptionNode(strike={self._strike}, price={self._price}, expiration={self._expiration})")
    
class SkewList:

    # This is a class specifically used for OptionGraph's get_skew() method

    def __init__(self, option_node_list: list[OptionNode]) -> None:
        self._option_node_list: list[OptionNode] = option_node_list

    def __getitem__(self, index):

        if isinstance(index, slice): return SkewList(self._option_node_list[index])

        else: return self._option_node_list[index]

    def implied_volatilities(self) -> list[float]:
        return [float(op.get_implied_volatility()) for op in self._option_node_list]
    
    def strikes(self) -> list[float]:
        return [float(op.get_strike()) for op in self._option_node_list]
    
    def prices(self) -> list[float]:
        return [float(op.get_price()) for op in self._option_node_list]
    
    def get_dte(self) -> int:
        return self._option_node_list[0].get_dte()
    
class TermStructureList:

    # This is a class specifically used for OptionGraph's get_term_structure() method

    def __init__(self, option_node_list: list[OptionNode]) -> None:
        self._option_node_list: list[OptionNode] = option_node_list

    def implied_volatilities(self) -> list[float]:
        return [float(op.get_implied_volatility()) for op in self._option_node_list]
    
    def expirations(self) -> list[float]:
        return [float(op.get_expiration()) for op in self._option_node_list]
    
    def strikes(self) -> list[float]:
        return [float(op.get_strike()) for op in self._option_node_list]
    
    def prices(self) -> list[float]:
        return [float(op.get_price()) for op in self._option_node_list]
    
class OptionGraph:

    def __init__(self):
        self.nodes: dict = {}
        self.expiration_index: dict = {}
        self.strike_index: dict = {}

    def add_option(self, option_node: OptionNode):

        key = (option_node._expiration, option_node._strike)
        if key not in self.nodes:
            self.nodes[key] = option_node
            
            if option_node._expiration not in self.expiration_index:
                self.expiration_index[option_node._expiration] = []
            self.expiration_index[option_node._expiration].append(option_node)

            if option_node._strike not in self.strike_index:
                self.strike_index[option_node._strike] = []
            self.strike_index[option_node._strike].append(option_node)

    def get_option(self, expiration: str, strike: int) -> OptionNode:

        return self.nodes.get((expiration, strike))
    
    def get_expirations(self) -> list[str]:

        return self.expiration_index.keys()
    
    def get_skew(self, expiration: str) -> list[OptionNode]:

        skew = SkewList(self.expiration_index[expiration])

        return skew
    
    def get_term_structure(self, strike: int) -> list[OptionNode]:

        term_structure = TermStructureList(self.strike_index[strike])

        return term_structure
    
    def __repr__(self):
        return f"OptionGraph"