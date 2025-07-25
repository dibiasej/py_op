from abc import ABC, abstractmethod
import datetime as dt

from utils import date_utils
from utils import util_funcs

TODAY: str = dt.datetime.now().strftime('%Y-%m-%d')

class OptionNode(ABC):

    """class that acts as a data hub for our option and stores all our values corresponding to the specific option"""

    def __init__(self, strike: int = None, price: float = None, bid: float = None, ask: float = None, implied_volatility: float = None,
                expiration: str = None, dte: int = None, volume: int = None, open_interest: int = None, otype: str = None) -> None:

        self._strike: int = strike
        self._price: float = price
        self._mid_price: float = (bid + ask)/2
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
    
    def get_mid_price(self) -> float:
        return self._mid_price
    
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
        self._mid_price: float = (bid + ask)/2
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

    def __len__(self) -> int:
        return len(self._option_node_list)
    
    def __iter__(self):
        for node in self._option_node_list:
            yield node
    
class TermStructureList:

    # This is a class specifically used for OptionGraph's get_term_structure() method

    def __init__(self, option_node_list: list[OptionNode]) -> None:
        self._option_node_list: list[OptionNode] = option_node_list

    def implied_volatilities(self) -> list[float]:
        return [float(op.get_implied_volatility()) for op in self._option_node_list]
    
    def expirations(self) -> list[float]:
        return [op.get_expiration() for op in self._option_node_list]
    
    def strikes(self) -> list[float]:
        return [float(op.get_strike()) for op in self._option_node_list]
    
    def prices(self) -> list[float]:
        return [float(op.get_price()) for op in self._option_node_list]
    
    def __iter__(self):
        for node in self._option_node_list:
            yield node
    
class OptionGraph:

    def __init__(self):
        self.nodes: dict = {}
        self.expiration_index: dict = {}
        self.strike_index: dict = {}
        self.close_date: str = None
        self.ticker = None

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
    
    def remove_option(self, exp: str, strike: float):

        key = (exp, strike)

        if key not in self.nodes:
            return
        
        option_node = self.nodes.pop((exp, strike))

        if exp in self.expiration_index:
            self.expiration_index[exp] = [
                node for node in self.expiration_index[exp]
                if node._strike != strike
            ]

        if strike in self.strike_index:
            self.strike_index[strike] = [
                node for node in self.strike_index[strike]
                if node._expiration != exp
            ]

    def get_option(self, expiration: str, strike: int) -> OptionNode:
        """
        Change this so it is a dunder method and we dont call get_option, also make a get closest option (by strike)
        """

        return self.nodes.get((expiration, strike))
    
    def get_expirations(self) -> list[str]:

        return self.expiration_index.keys()
    
    def get_expirations_from_int(self, exp_int: int) -> str:
        """
        Given a int representing a number of days return a str of the expiration closest to that number of days.
        We use fetch_exp() function imported from date utils
        Params:
        int exp_int: days until expiration represented as a integer.
        Return:
        str : epiration data format "YYYY-MM-DD" 
        """
        return date_utils.fetch_exp(self, exp_int)
    
    def get_expirations_from_str(self, exp_str: str) -> str:
        """
        Given a str representing a expiration date ex("1M", "2M') return a str of the expiration closest to that str.
        We use fetch_exp() function imported from date utils
        Params:
        str exp_str: days until expiration represented as a str.
        Return:
        str : epiration data format "YYYY-MM-DD" 
        """
        return date_utils.fetch_exp(self, exp_str)
    
    def get_dte_from_str(self, exp_str: str) -> float:
        """
        Given a string of format YYYY-MM-DD return a float representing dte
        Params:
        str exp_str: days until expiration represented as a str.
        Return:
        float: numeric dte.
        """
        return self.get_skew(exp_str).get_dte()
    
    def get_specified_strikes(self, min_strike: float, max_strike: float, steps: int = 1, exp: str = None) -> list[float]:
        """
        This method gives us a list of strikes based on our specified criteria being strike max, min, expiration of the strike list and steps between strikes.
        Params:
        min_strike: minimum strike in the list.
        max_strike: max strike in the list.
        steps: space of distance between adjacent strikes.
        exp: either None which then grabs the list of expirations from the graph data structure and uses the second element 
        """

        if exp is None:
            exps = list(self.get_expirations())
            skew = self.get_skew(exps[1])
        
        else:
            skew = self.get_skew(exp)

        strike_arr = skew.strikes()
        strike_range = [min_strike, max_strike]

        strikes = [strike for strike in strike_arr if strike_range[0] <= strike <= strike_range[1] and strike % steps == 0]

        return strikes
    
    def get_skew(self, expiration: str) -> list[OptionNode]:
        """
        Params:
        str exp: Expiration date represented as "YYYY-MM-DD"
        """
        # might need to add a check for correct format
        skew = SkewList(self.expiration_index[expiration])

        return skew
    
    def get_term_structure(self, strike: int) -> list[OptionNode]:

        term_structure = TermStructureList(self.strike_index[strike])

        return term_structure
    
    def get_atm_option(self, exp: str, close_date: str = TODAY, atm_price: float = None) -> OptionNode:
        return util_funcs.get_atm_option(self, exp, close_date, atm_price)
    
    def __iter__(self):
        """
        This special method is used so we can loop through our OptionGraph data structure.
        We loop by expiration, so every OptionNode for a certain expiration is returned one at a time, then the next expiration is looped through and etc..
        """
        for exp in self.expiration_index:
            for node in self.expiration_index[exp]:
                yield node
        
    
    #def __next__(self):

    
    def __repr__(self):
        return f"OptionGraph"