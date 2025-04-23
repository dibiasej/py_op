import numpy as np
from abc import abstractmethod, ABC

from data.price_data import process_price_data
#from realized_volatility import realized_volatility_utils as rv_utils
from . import realized_volatility_utils as rv_utils


"""
In this model we have multiple different methods for calculating realized volatility
Note: This module is heavily dependent on process_price_data and yahoo finance to get and clean the data so if this breaks look into those
Note: We still need to add EWMA, GARCH, etc... 
Note: We should move this into calc engine but make a seperate folder called realized vol inside the volatility folder
"""
class HistoricalVolatilityStrategy(ABC):
    @abstractmethod
    def calculate(ticker: str, start: str, end: str = None, realized_volatility_period: str = "M", freq: str = "D") -> list[float]:
        pass

class CloseToClose(HistoricalVolatilityStrategy):
    def calculate(self, ticker: str, start: str, end: str = None, realized_volatility_period: str = "M", freq: str = "D") -> list[float]:

        t: float = rv_utils._t(freq)
        length_int = rv_utils.realized_volatility_period_length(realized_volatility_period, freq)

        log_rets = process_price_data.get_log_rets(ticker, start, end, freq)
        log_rets = log_rets[::-1]

        rvol: list = [np.sqrt(sum(log_rets[0 + i: length_int + i]**2) / length_int) * np.sqrt(t) for i in range(len(log_rets) - length_int)]
        return rvol[::-1]
    
    def __str__(self):
        return "Close to Close"

class Parkinsons(HistoricalVolatilityStrategy):
    def calculate(self, ticker: str, start: str, end: str = None, realized_volatility_period: str = "M", freq: str = "D") -> list[float]:

        t: float = rv_utils._t(freq)
        length_int: int = rv_utils.realized_volatility_period_length(realized_volatility_period, freq)

        high_low: list[float] = np.array(process_price_data.get_high_low_rets(ticker, start, end, freq)) ** 2
        high_low = high_low[::-1]
        
        parkinson = np.array([np.sqrt((1 / (4 * np.log(2))) * sum(high_low[0 + i: length_int + i])) for i in range(len(high_low) - length_int)])[::-1]
        
        return parkinson * np.sqrt(t / length_int)
    
    def __str__(self):
        return "Parkinsons"

class RogerSatchell(HistoricalVolatilityStrategy):
    def calculate(self, ticker: str, start: str, end: str = None, realized_volatility_period: str = "M", freq: str = "D") -> list[float]:

        t: float = rv_utils._t(freq)
        length_int: int = rv_utils.realized_volatility_period_length(realized_volatility_period, freq)

        high_close: list[float] = np.array(process_price_data.get_high_close_rets(ticker, start, end, freq))
        high_open: list[float] = np.array(process_price_data.get_high_open_rets(ticker, start, end, freq))
        low_close: list[float] = np.array(process_price_data.get_low_close_rets(ticker, start, end, freq))
        low_open: list[float] = np.array(process_price_data.get_low_open_rets(ticker, start, end, freq))

        rs = high_close*high_open + low_close*low_open
        rs = rs[::-1]

        roger_satchell = np.array([sum(rs[0 + i: length_int + i]) for i in range(len(rs) - length_int)])[::-1]

        return np.sqrt(roger_satchell / length_int) * np.sqrt(t)
    
    def __str__(self):
        return "Rogers Satchell"

class GarmanKlass(HistoricalVolatilityStrategy):
    def calculate(self, ticker: str, start: str, end: str = None, realized_volatility_period: str = "M", freq: str = "D") -> list[float]:

        t: float = rv_utils._t(freq)
        length_int: int = rv_utils.realized_volatility_period_length(realized_volatility_period, freq)

        high_low: list[float] = np.array(process_price_data.get_high_low_rets(ticker, start, end, freq)) ** 2
        close_open: list[float] = np.array(process_price_data.get_close_open_rets(ticker, start, end, freq)) ** 2

        gk = .5*high_low - (2 * np.log(2) - 1) * (close_open)
        gk = gk[::-1]

        garman_klass = np.sqrt((1 / length_int) * np.array([sum(gk[0 + i: length_int + i]) for i in range(len(high_low) - length_int)]))[::-1]

        return garman_klass * np.sqrt(t)
    
    def __str__(self):
        return "Garman Klass"

class YangZhang(HistoricalVolatilityStrategy):
    def calculate(self, ticker: str, start: str, end: str = None, realized_volatility_period: str = "M", freq: str = "D") -> list[float]:

        t: float = rv_utils._t(freq)
        length_int: int = rv_utils.realized_volatility_period_length(realized_volatility_period, freq)

        high_low: list[float] = np.array(process_price_data.get_high_low_rets(ticker, start, end, freq)) ** 2
        close_open: list[float] = np.array(process_price_data.get_close_open_rets(ticker, start, end, freq)) ** 2
        open_close: list[float] = np.array(process_price_data.get_open_close_rets(ticker, start, end, freq)) ** 2

        yz: list[float] = open_close + .5*high_low[:len(high_low) - 1] - (2 * np.log(2) - 1) * (close_open[:len(close_open) - 1])
        yz = yz[::-1]

        yang_zhang: list[float] = np.sqrt((1 / length_int) * np.array([sum(yz[0 + i: length_int + i]) for i in range(len(high_low) - length_int)]))[::-1]

        return yang_zhang * np.sqrt(t)
    
    def __str__(self):
        return "Yang Zhang"