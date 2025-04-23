import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod

class SkewParameterization:
    def __init__(self, implied_volatilities: list[float], strikes: list[float]) -> None:
        self._implied_volatilities: list[float] = implied_volatilities
        self._strikes: list[float] = strikes
    
    def delta(self, delta_list: list[float]):
        pass
    
    def factset(self, S: float, otype: str):

        if otype == "call":
            func: str = f"iv > .05 and iv < 2 and strike > {S}"

        elif otype == "put":
            func: str = f"iv > .05 and iv < 2 and strike < {S}"

        filtered_ivs_strikes: list[tuple(float, float)] = [(iv, strike) for iv, strike in zip(self._implied_volatilities, self._strikes) if eval(func)]
        self._implied_volatilities, self._strikes = zip(*filtered_ivs_strikes)

        return list(self._implied_volatilities), list(self._strikes)
    
class SkewParameterization:
    pass

class SkewXAxis(SkewParameterization):

    @abstractmethod
    def filter(self, logic) -> None:
        pass

class SkewYAxis(SkewParameterization):

    @abstractmethod
    def filter(self, logic) -> None:
        pass

class SkewMoneyness(SkewXAxis):
    
    def __init__(self, implied_volatilities: list[float], strikes: list[float]) -> None:
        self._implied_volatilities: list[float] = implied_volatilities
        self._strikes: list[float] = strikes

    def filter(self, logic) -> None:
        filtered_ivs_strikes: list[tuple(float, float)] = [(iv, strike) for iv, strike in zip(self._implied_volatilities, self._strikes) if eval(logic)]
        self._implied_volatilities, self._strikes = zip(*filtered_ivs_strikes)

    def moneyness_parameterized(self, S: float) -> (list[float], list[float]):

        # try to find a way to get the atm strike so we dont need to pass in S

        filtered_ivs_strikes: list[tuple(float, float)] = [(iv, np.log(strike / S)) for iv, strike in zip(self._implied_volatilities, self._strikes)]# if eval(func)]
        ivs, strikes = zip(*filtered_ivs_strikes)

        return list(ivs), list(strikes)
    
    def moneyness_parameterized_recal(self, S: float, T: float) -> (list[float], list[float]):
        # change the name for this
        # try to find a way to get the atm strike so we dont need to pass in S
        if T > 1:
            T /= 365

        atm_iv = [iv for iv, strike in zip(self._implied_volatilities, self._strikes) if strike <= S]

        filtered_ivs_strikes: list[tuple(float, float)] = [(iv, np.log(strike / S) / (atm_iv[-1] * np.sqrt(T))) for iv, strike in zip(self._implied_volatilities, self._strikes)]# if eval(func)]
        ivs, strikes = zip(*filtered_ivs_strikes)

        return list(ivs), list(strikes)

class SkewStrikeParameterization:
    def __init__(self, implied_volatilities: list[float], strikes: list[float]) -> None:
        self._implied_volatilities: list[float] = implied_volatilities
        self._strikes: list[float] = strikes

    def filter(self, logic) -> None:
        filtered_ivs_strikes: list[tuple(float, float)] = [(iv, strike) for iv, strike in zip(self._implied_volatilities, self._strikes) if eval(logic)]
        self._implied_volatilities, self._strikes = zip(*filtered_ivs_strikes)

    def moneyness_parameterized(self, S: float) -> (list[float], list[float]):

        # try to find a way to get the atm strike so we dont need to pass in S

        filtered_ivs_strikes: list[tuple(float, float)] = [(iv, np.log(strike / S)) for iv, strike in zip(self._implied_volatilities, self._strikes)]# if eval(func)]
        ivs, strikes = zip(*filtered_ivs_strikes)

        return list(ivs), list(strikes)
    
    def moneyness_parameterized_recal(self, S: float, T: float) -> (list[float], list[float]):
        # change the name for this
        # try to find a way to get the atm strike so we dont need to pass in S
        if T > 1:
            T /= 365

        atm_iv = [iv for iv, strike in zip(self._implied_volatilities, self._strikes) if strike <= S]

        filtered_ivs_strikes: list[tuple(float, float)] = [(iv, np.log(strike / S) / (atm_iv[-1] * np.sqrt(T))) for iv, strike in zip(self._implied_volatilities, self._strikes)]# if eval(func)]
        ivs, strikes = zip(*filtered_ivs_strikes)

        return list(ivs), list(strikes)
      
class SkewDeltaParameterization:
    def __init__(self, implied_volatilities: list[float], deltas: list[float] ) -> None:
        self._implied_volatilities: list[float] = implied_volatilities
        self._deltas: list[float] = deltas

    def filter(self):
        pass

    def bin(self):
        pass

class SkewAltParameterization:
    # We will put Factset in here
    pass
    
class SkewFactset:

    def __init__(self, call_implied_volatilities: list[float], put_implied_volatilities: list[float], call_strikes: list[float], put_strikes: list[float]) -> None:
        self._call_implied_volatilities: list[float] = call_implied_volatilities
        self._put_implied_volatilities: list[float] = put_implied_volatilities
        self._call_strikes: list[float] = call_strikes
        self._put_strikes: list[float] = put_strikes
        self._call_skew: SkewParameterization = None
        self._put_skew: SkewParameterization = None

    def get_data(self, S: float) -> (list[float], list[float]):
        if self._call_skew is None:
            self._call_skew = SkewParameterization(self._call_implied_volatilities, self._call_strikes)
        call_ivs, call_strikes = self._call_skew.factset(S, 'call')

        if self._put_skew is None:
            self._put_skew = SkewParameterization(self._put_implied_volatilities, self._put_strikes)
        put_ivs, put_strikes = self._put_skew.factset(S, 'put')

        strike_list = put_strikes + call_strikes

        atm_put_iv: float = put_ivs[-1]
        #avg_put_call_iv: float = sum(put_iv_list + call_iv_list) / len(put_iv_list + call_iv_list)
        avg_put_call_iv: float = (put_ivs[-1] + call_ivs[0]) / 2
        adj_factor: float = atm_put_iv - avg_put_call_iv

        adj_call_iv = [iv + adj_factor for iv in call_ivs]
        adj_put_iv = [iv - adj_factor for iv in put_ivs]

        adj_iv_list = adj_put_iv + adj_call_iv

        return adj_iv_list, strike_list
    
def plot_skew(skew_structure: SkewParameterization, func: str = "iv > .02"):

    ivs1, strikes1 = skew_structure.filter(func)

    print(func)
    print(len(strikes1))

    plt.plot(strikes1, ivs1, '.')
    plt.show()
    return None