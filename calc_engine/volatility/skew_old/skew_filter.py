import numpy as np
from abc import ABC, abstractmethod

class Filter:
    @abstractmethod
    def filter(self, x_data, y_data: list[float], *args, **kwargs) -> list[float]:
        pass

class Strike(Filter):
    pass

class Moneyness(Filter):
    """
    Class for parameterizing iv skew x axis
    We must pass in a value for spot like this S = xx.xx
    """

    def filter(self, implied_volatilities: list[float], strikes: list[float], **kwargs) -> (list[float], list[float], list[float]):
        if "S" in kwargs:
            S = kwargs["S"]
        else:
            return ValueError("Keyword argument must have the parameter S")
        
        if "filter_logic" in kwargs:
        
            logic = kwargs["filter_logic"]

        else:
            logic = "iv > .05"
        
        moneyness_ivs_strikes: list[tuple(float, float, float)] = [(iv, x, (x / S)) for iv, x in zip(implied_volatilities, strikes) if eval(logic)]
        ivs, strikes, moneyness = zip(*moneyness_ivs_strikes)

        return list(ivs), list(strikes), list(moneyness)
    
class LogMoneyness(Filter):
    """
    Class for parameterizing iv skew x axis
    We must pass in a value for spot like this S = xx.xx
    """
    def filter(self, implied_volatilities: list[float], strikes: list[float], **kwargs) -> (list[float], list[float], list[float]):
        if "S" in kwargs:
            S = kwargs["S"]
        else:
            return ValueError("Keyword argument must have the parameter S")
        
        if "filter_logic" in kwargs:
        
            logic = kwargs["filter_logic"]

        else:
            logic = "iv > .05"
        
        filtered_ivs_strikes: list[tuple(float, float, float)] = [(iv, x, np.log(x / S)) for iv, x in zip(implied_volatilities, strikes) if eval(logic)]
        ivs, strikes, log_moneyness = zip(*filtered_ivs_strikes)

        return list(ivs), list(strikes), list(log_moneyness)
    
class LogMoneynessTime(Filter):
    """
    Class for parameterizing iv skew x axis
    We must pass in a value for spot like this S = xx.xx
    """
    def filter(self, implied_volatilities: list[float], strikes: list[float], **kwargs) -> (list[float], list[float], list[float]):
        if "S" in kwargs:
            S = kwargs["S"]
        else:
            return ValueError("Keyword argument must have the parameter S")
        
        if "dte" in kwargs:
            dte = kwargs["dte"]

            if dte > 1:
                dte /= 365

        else:
            return ValueError("Keyword argument must have the parameter dte")
        
        if "filter_logic" in kwargs:
        
            logic = kwargs["filter_logic"]

        else:
            logic = "iv > .05"
        
        filtered_ivs_strikes: list[tuple(float, float, float)] = [(iv, x, np.log(x / S) / np.sqrt(dte)) for iv, x in zip(implied_volatilities, strikes) if eval(logic)]
        ivs, strikes, log_moneyness_time = zip(*filtered_ivs_strikes)

        return list(ivs), list(strikes), list(log_moneyness_time)
    
class SkewParameterization:

    def __init__(self, x_filter_strategy: Filter) -> None:

        self._x_filter_strategy: Filter = x_filter_strategy

    def filter(self, implied_volatilities: list[float], x_data: list[float], **kwargs) -> tuple:

        ivs, strikes, filtered_data = self._x_filter_strategy.filter(implied_volatilities, x_data, **kwargs)

        return ivs, strikes, filtered_data