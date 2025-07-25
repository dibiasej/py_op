import numpy as np

from data.data_processor.data_processor import MarketPriceProcessor
from calc_engine.volatility.iv_calc import ImpliedVolatility
from calc_engine.calibration.put_call_parity import implied_rate
"""
module for term structures
We will get rid of MarketTermStructure and add it to data_processor.py
"""
class MarketTermStructure:

    def __init__(self, ticker: str, close_date: str, S: float) -> None:
        self.ticker: str = ticker
        self.close_date: str = close_date
        self.S: float = S
        self.market_processor = MarketPriceProcessor(self.ticker, self.close_date)

        self.exps_str: list[str] = ['1M', '2M', '3M', '6M', '9M', '1Y', '2Y']
        self.call_exps = [self.market_processor.option_call_graph.get_expirations_from_str(exp) for exp in self.exps_str]
        self.put_exps = [self.market_processor.option_put_graph.get_expirations_from_str(exp) for exp in self.exps_str]
        self.dtes = np.array([self.market_processor.option_call_graph.get_dte_from_str(exp)/365 for exp in self.call_exps])

        self.F: list[float] = self.S * np.exp(.04*self.dtes)

        self.atm_put_strikes, self.atm_put_prices = zip(*[self.market_processor.atm_put_option(self.S, exp) for exp in self.put_exps])
        self.atm_call_strikes, self.atm_call_prices = zip(*[self.market_processor.atm_call_option(self.S, exp) for exp in self.put_exps])

        self.atf_put_strikes, self.atf_put_prices = zip(*[self.market_processor.atm_put_option(f, exp) for exp, f in zip(self.put_exps, self.F)])
        self.atf_call_strikes, self.atf_call_prices = zip(*[self.market_processor.atm_call_option(f, exp) for exp, f in zip(self.put_exps, self.F)])

        self.implied_rates = [implied_rate(call_price, put_price, self.S, self.S, dte) for call_price, put_price, dte in zip(self.atm_call_prices, self.atm_put_prices, self.dtes)]
        self.implied_rates_forward = [implied_rate(call_price_f, put_price_f, f, (put_strike_f + call_strike_f)/2, dte) for call_price_f, put_price_f, dte, f, put_strike_f, call_strike_f in zip(self.atm_call_prices, self.atm_put_prices, self.dtes, self.F, self.atf_put_strikes, self.atf_call_strikes)]

    def atm_put(self) -> (list[str], list[float]):
        atm_put_ivs = [float(ImpliedVolatility().root_finder(atm_price, self.S, atm_strike, dte, r=rate, otype='put')) for atm_price, atm_strike, dte, rate in zip(self.atm_put_prices, self.atm_put_strikes, self.dtes, self.implied_rates)]
        return self.dtes, atm_put_ivs
    
    def atm_call(self) -> (list[str], list[float]):
        atm_call_ivs = [float(ImpliedVolatility().root_finder(atm_price, self.S, atm_strike, dte, r=rate, otype='call')) for atm_price, atm_strike, dte, rate in zip(self.atm_call_prices, self.atm_call_strikes, self.dtes, self.implied_rates)]
        return self.dtes, atm_call_ivs
    
    def atf_put(self) -> (list[str], list[float]):
        atf_put_ivs = [float(ImpliedVolatility().root_finder(atf_price, f, atf_strike, dte, r=rate, otype='put')) for atf_price, atf_strike, dte, rate, f in zip(self.atf_put_prices, self.atf_put_strikes, self.dtes, self.implied_rates_forward, self.F)]
        return self.dtes, atf_put_ivs
      
    def atf_call(self) -> (list[str], list[float]):
        atf_call_ivs = [float(ImpliedVolatility().root_finder(atf_price, f, atf_strike, dte, r=rate, otype='call')) for atf_price, atf_strike, dte, rate, f in zip(self.atf_call_prices, self.atf_call_strikes, self.dtes, self.implied_rates_forward, self.F)]
        return self.dtes, atf_call_ivs
          
    def atm_put_delta(self) -> (list[str], list[float], list[float]):
        strikes, prices, ivs = zip(*[self.market_processor.delta_option(self.S, -.5, exp) for exp in self.put_exps])
        return self.dtes, prices, ivs
    
    def atm_call_delta(self) -> (list[str], list[float], list[float]):
        strikes, prices, ivs = zip(*[self.market_processor.delta_option(self.S, .5, exp) for exp in self.call_exps])
        return self.dtes, prices, ivs
    
def realized_vol():
    pass

def forward_vol(ivs: list[float], dtes: list[float]):
    ivs, dtes = np.array(ivs), np.array(dtes)
    forward_ivs = []
    forward_dtes = []
    
    for i in range(len(ivs) - 1):
        forward_iv = np.sqrt((dtes[i + 1] * ivs[i + 1]**2 - dtes[i] * ivs[i]**2) / (dtes[i + 1] - dtes[i]))
        forward_ivs.append(forward_iv)

        forward_dte = f"{dtes[i + 1]*365} - {dtes[i]*365}"
        forward_dtes.append(forward_dte)

    return forward_dtes, forward_ivs