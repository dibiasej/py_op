import numpy as np

from data.option_data.process_option_chain import OptionFactory
from calc_engine.volatility.iv_calc import ImpliedVolatility
from calc_engine.greeks.analytical_greeks import AnalyticalDelta
from utils.util_funcs import get_stock_price
from calc_engine.calibration.put_call_parity import implied_rate

"""
Need to make a method or function that takes in two array one put one call, they should also be lists with tuples of corresponding strikes.
EX: puts = [(K, model price), (K, model price), (K, model price), (K, model price), (K, model price), (K, model price)]
    calls = [(K, model price), (K, model price), (K, model price), (K, model price), (K, model price), (K, model price)]
    then we make a function given these two lists and a spot price and make a loop using a itertools function to look and compare with a 
    condition saying if K < S use puts else use calls model price

Note: 7/15/2025 - We are changing all the option_call_graph.get_option(exp, strike).get_price() to option_call_graph.get_option(exp, strike).get_mid_price()
                - For both puts and calls
"""

class MarketPriceProcessor:

    def __init__(self, ticker: str, close_date: str):
        self.option_call_graph = OptionFactory().create_option_graph(ticker, close_date, option_type = 'call')
        self.option_put_graph = OptionFactory().create_option_graph(ticker, close_date, option_type = 'put')
    
    def otm_call(self, S: float, max_strike: float, exp: str, steps: int = 1) -> list[float]:

        strikes = np.array(self.option_call_graph.get_skew(exp).strikes())
        idxs = np.where((strikes > S) & (strikes <= max_strike))[0]
        strikes = strikes[idxs]

        #return [(float(strike), float(self.option_call_graph.get_option(exp, strike).get_price())) for strike in strikes if strike % steps == 0]
        return [(float(strike), float(self.option_call_graph.get_option(exp, strike).get_mid_price())) for strike in strikes if strike % steps == 0]
    
    def otm_put(self, S: float, min_strike: float, exp: str, steps: int = 1) -> list[float]:

        strikes = np.array(self.option_put_graph.get_skew(exp).strikes())
        idxs = np.where((strikes < S) & (strikes >= min_strike))[0]
        strikes = strikes[idxs]

        #return [(float(strike), float(self.option_put_graph.get_option(exp, strike).get_price())) for strike in strikes if strike % steps == 0]
        return [(float(strike), float(self.option_put_graph.get_option(exp, strike).get_mid_price())) for strike in strikes if strike % steps == 0]
    
    def all_calls(self, min_strike: float, max_strike: float, exp: str, steps: int = 1) -> list[float]:
        strikes = np.array(self.option_call_graph.get_skew(exp).strikes())
        idxs = np.where((min_strike <= strikes) & (strikes <= max_strike))[0]
        strikes = strikes[idxs]

        #return [(float(strike), float(self.option_call_graph.get_option(exp, strike).get_price())) for strike in strikes if strike % steps == 0]
        return [(float(strike), float(self.option_call_graph.get_option(exp, strike).get_mid_price())) for strike in strikes if strike % steps == 0]
    
    def all_puts(self,  min_strike: float, max_strike: float, exp: str, steps: int = 1) -> list[float]:
        strikes = np.array(self.option_put_graph.get_skew(exp).strikes())
        idxs = np.where((strikes <= max_strike) & (strikes >= min_strike))[0]
        strikes = strikes[idxs]

        #return [(float(strike), float(self.option_put_graph.get_option(exp, strike).get_price())) for strike in strikes if strike % steps == 0]
        return [(float(strike), float(self.option_put_graph.get_option(exp, strike).get_mid_price())) for strike in strikes if strike % steps == 0]
    
    def atm_option(self, S: float, exp: str) -> float:
        
        call_strikes = np.array(self.option_call_graph.get_skew(exp).strikes())
        put_strikes = np.array(self.option_put_graph.get_skew(exp).strikes())

        call_idx = np.abs(call_strikes - S).argmin()
        put_idx = np.abs(put_strikes - S).argmin()

        call_strike = call_strikes[call_idx]
        put_strike = put_strikes[put_idx]

        #atm_call_price = self.option_call_graph.get_option(exp, call_strike).get_price()
        #atm_put_price = self.option_put_graph.get_option(exp, put_strike).get_price()
        atm_call_price = self.option_call_graph.get_option(exp, call_strike).get_mid_price()
        atm_put_price = self.option_put_graph.get_option(exp, put_strike).get_mid_price()

        return (float((call_strike + put_strike) / 2), float((atm_call_price + atm_put_price) / 2))
    
    def atm_call_option(self, S: float, exp: str) -> float:
        call_strikes = np.array(self.option_call_graph.get_skew(exp).strikes())

        call_idx = np.abs(call_strikes - S).argmin()

        call_strike = call_strikes[call_idx]

        #atm_call_price = self.option_call_graph.get_option(exp, call_strike).get_price()
        atm_call_price = self.option_call_graph.get_option(exp, call_strike).get_mid_price()

        return (float(call_strike), float(atm_call_price))
    
    def atm_put_option(self, S: float, exp: str) -> float:
        
        put_strikes = np.array(self.option_put_graph.get_skew(exp).strikes())

        put_idx = np.abs(put_strikes - S).argmin()

        put_strike = put_strikes[put_idx]

        #atm_put_price = self.option_put_graph.get_option(exp, put_strike).get_price()
        atm_put_price = self.option_put_graph.get_option(exp, put_strike).get_mid_price()

        return (float(put_strike), float(atm_put_price))
    
    def K_option(self, S: float, K: float, exp: float):
        if S < K:
           call_strikes = np.array(self.option_call_graph.get_skew(exp).strikes()) 
           call_idx = np.abs(call_strikes - K).argmin()
           call_strike = call_strikes[call_idx]
           #call_price = self.option_call_graph.get_option(exp, call_strike).get_price()
           call_price = self.option_call_graph.get_option(exp, call_strike).get_mid_price()
           return (call_strike, call_price)
        
        if S > K:
           put_strikes = np.array(self.option_put_graph.get_skew(exp).strikes()) 
           put_idx = np.abs(put_strikes - K).argmin()
           put_strike = put_strikes[put_idx]
           #put_price = self.option_put_graph.get_option(exp, put_strike).get_price()
           put_price = self.option_put_graph.get_option(exp, put_strike).get_mid_price()
           return (put_strike, put_price)
        
    def moneyness_option(self, S, moneyness, exp):
    
        if moneyness < 1:
            put_strikes = np.array(self.option_put_graph.get_skew(exp).strikes())
            idx = np.abs(put_strikes/S - moneyness).argmin()
            put_strike = put_strikes[idx]
            #put_price = self.option_put_graph.get_option(exp, put_strike).get_price()
            put_price = self.option_put_graph.get_option(exp, put_strike).get_mid_price()
            return (float(put_strike), float(put_price))
        
        elif moneyness > 1:
            call_strikes = np.array(self.option_call_graph.get_skew(exp).strikes())
            idx = np.abs(call_strikes/S - moneyness).argmin()
            call_strike = call_strikes[idx]
            #call_price = self.option_call_graph.get_option(exp, call_strike).get_price()
            call_price = self.option_call_graph.get_option(exp, call_strike).get_mid_price()
            return (float(call_strike), float(call_price))
        
        elif moneyness == 1:
            return self.atm_option(S, exp)
        
    def delta_option(self, S, delta, exp, r = 0.05, steps = 1):

        call_strikes_orig = np.array(self.option_call_graph.get_skew(exp).strikes())
        put_strikes_orig = np.array(self.option_put_graph.get_skew(exp).strikes())
        dte = self.option_call_graph.get_dte_from_str(exp)/252

        if delta > 0:
            call_idxs = np.where((call_strikes_orig > S))[0]
            call_strikes = call_strikes_orig[call_idxs]
            #call_prices = [float(self.option_call_graph.get_option(exp, strike).get_price()) for strike in call_strikes if strike % steps == 0]
            call_prices = [float(self.option_call_graph.get_option(exp, strike).get_mid_price()) for strike in call_strikes if strike % steps == 0]
            call_ivs = [ImpliedVolatility().root_finder(call_prices[i], S, call_strikes[i], dte, r = r, otype='call') for i in range(len(call_prices))]

            new_call_strikes, call_deltas = zip(*[(call_strikes[i], AnalyticalDelta().calculate(S, call_strikes[i], dte, call_ivs[i], r=r, otype='call')) for i in range(len(call_prices))])
            new_call_strikes, call_deltas = np.array(new_call_strikes), np.array(call_deltas)
            delta_call_idx = np.abs(call_deltas - delta).argmin()
            delta_call_strike = call_strikes[delta_call_idx]
            delta_call_price = call_prices[delta_call_idx]
            delta_call_iv = call_ivs[delta_call_idx]
            return (float(delta_call_strike), delta_call_price, float(delta_call_iv))

        if delta < 0:
            put_idxs = np.where((put_strikes_orig < S))[0]
            put_strikes = put_strikes_orig[put_idxs]
            #put_prices = [float(self.option_put_graph.get_option(exp, strike).get_price()) for strike in put_strikes if strike % steps == 0]
            put_prices = [float(self.option_put_graph.get_option(exp, strike).get_mid_price()) for strike in put_strikes if strike % steps == 0]
            put_ivs = [ImpliedVolatility().root_finder(put_prices[i], S, put_strikes[i], dte, r = r, otype='put') for i in range(len(put_prices))]

            new_put_strikes, put_deltas = zip(*[(put_strikes[i], AnalyticalDelta().calculate(S, put_strikes[i], dte, put_ivs[i], r=r, otype='put')) for i in range(len(put_prices))])
            new_put_strikes, put_deltas = np.array(new_put_strikes), np.array(put_deltas)
            delta_put_idx = np.abs(put_deltas - delta).argmin()
            delta_put_strike = put_strikes[delta_put_idx]
            delta_put_price = put_prices[delta_put_idx]
            delta_put_iv = put_ivs[delta_put_idx]
            return (float(delta_put_strike), delta_put_price, float(delta_put_iv))
        
    def put_call_price_skew(self, S: float, min_strike: float, max_strike: float, exp: str, steps: int = 1):
        #return self.otm_put(S, min_strike, exp, steps) + [self.atm_option(S, exp)] + self.otm_call(S, max_strike, exp, steps)
        #return self.otm_put(S, min_strike, exp, steps) + [self.atm_put_option(S, exp)] + [self.atm_call_option(S, exp)] + self.otm_call(S, max_strike, exp, steps)
        return self.otm_put(S, min_strike, exp, steps) + self.otm_call(S, max_strike, exp, steps)

    def put_call_price_surface(self, S: float, min_strike: list[float], max_strike: list[float], exps: list[str], steps: int = 1) -> list[list[float]]:
        """
        This returns a list of array where each array has tuples of strike and option price
        Each array in it will be of different length, when we plot our vol surfaces we must interploate in a way to make all len even.
        """
        return [self.put_call_price_skew(S, min_strike, max_strike, exp, steps) for exp in exps]
    
class ModelPriceProcessor:
    """
    Class to process model price data for options.
    Add two methods to grab list of itm and otm options, one for calls one for puts.
    Variance Gamma is not yet compatible with this class
    """

    def __init__(self, model: str) -> None:
        """
        model is a option pricing model from calc engine that had methods put and call
        """
        self.model = model
    
    def otm_call(self, S: float, strikes: np.ndarray, T: float, **kwargs) -> list[tuple[float, float]]:
        """
        Method that returns otm call prices
        """
        strikes = np.array(strikes)
        call_indices = np.where(strikes > S)[0]
        call_strikes = strikes[call_indices]
        model_call = self.model.call(S, call_strikes, T, **kwargs)
        call_results = list(zip(call_strikes, model_call))
        return [(float(strike), float(price)) for strike, price in call_results]
    
    def otm_put(self, S: float, strikes: np.ndarray, T: float, **kwargs) -> list[tuple[float, float]]:
        """
        Method that returns otm put prices
        """
        strikes = np.array(strikes)
        put_indices = np.where(strikes < S)[0]
        put_strikes = strikes[put_indices]
        model_put = self.model.put(S, put_strikes, T, **kwargs) 
        put_results = list(zip(put_strikes, model_put))
        return [(float(strike), float(price)) for strike, price in put_results]
    
    def atm_option(self, S: float, T: float, **kwargs) -> list[tuple[float, float]]:
        """
        Method that returns atm option price
        """
        model_call = self.model.call(S, S, T, **kwargs)
        model_put = self.model.put(S, S, T, **kwargs)
        return (model_call + model_put) / 2
    
    def put_call_price_skew(self, S: float, strikes: np.ndarray, T: float, **kwargs) -> list[tuple[float, float]]:
        """
        Method that returns a list of otm puts, the atm price and otm calls per expiry 
        """
        otm_calls = self.otm_call(S, strikes, T, **kwargs)
        otm_puts = self.otm_put(S, strikes, T, **kwargs)
        atm = self.atm_option(S, T, **kwargs)
        return otm_puts[:-1] + [(S, float(atm))] + otm_calls
    
    def price_term_structure(self, S: float, dtes: list[float], K: float = None, **kwargs) -> list[float]:
        """
        Method that returns a list of prices that represents the term structure of model prices.
        Since we specify the strike the logic we use grabs the otm price, if S < K we have a put term structure
        if S > K we have a call term structure
        """
        if K is None:
            return self.atm_option(S, dtes, **kwargs)
        elif S > K:
            return self.model.put(S, K, dtes, **kwargs)
        elif S < K:
            return self.model.call(S, K, dtes, **kwargs)
        
    def put_call_price_surface(self, S: float, strike_matrix: list[list[float]], dtes: list[float], **kwargs) -> list[list[float]]:
        return [self.put_call_price_skew(S, strikes, t, **kwargs) for strikes, t in zip(strike_matrix, dtes)]
    
class VolatilityProcessor:  

    def otm_put_ivs(self, S: float, prices: np.ndarray, strikes: np.ndarray, T: float, **kwargs) -> list[tuple[float, float]]:
        """
        Method that returns otm put prices
        """
        strikes, prices = np.array(strikes), np.array(prices)
        put_indices = np.where(strikes < S)[0]
        put_strikes = strikes[put_indices]
        put_prices = prices[put_indices] 
        ivs = [(put_strikes[i], ImpliedVolatility().root_finder(put_prices[i], S, put_strikes[i], T, otype='put')) for i in range(len(put_prices))]
        return ivs

    def otm_call_ivs(self, S: float, prices: np.ndarray, strikes: np.ndarray, T: float, **kwargs) -> list[tuple[float, float]]:
        """
        Method that returns otm put prices
        """
        strikes, prices = np.array(strikes), np.array(prices)
        call_indices = np.where(strikes > S)[0]
        call_strikes = strikes[call_indices]
        call_prices = prices[call_indices] 
        ivs = [(call_strikes[i], ImpliedVolatility().root_finder(call_prices[i], S, call_strikes[i], T, otype='call')) for i in range(len(call_prices))]
        return ivs

    def atm_option_iv(self, S: float, prices: np.ndarray, strikes: np.ndarray, dte: float) -> float:

        atm_idx = np.abs(strikes - S).argmin()
        atm_strike = strikes[atm_idx]
        atm_price = prices[atm_idx]
    
        atm_put_iv = ImpliedVolatility().root_finder(atm_price, S, atm_strike, dte, otype='put')
        atm_call_iv = ImpliedVolatility().root_finder(atm_price, S, atm_strike, dte, otype='call')

        return (float(atm_strike), float((atm_call_iv + atm_put_iv) / 2))
    
    def otm_put_call_skew(self, S: float, prices: np.ndarray, strikes: np.ndarray, dte: float) -> list[float]:
        # keep the atm price because it helps us fit skew better
        #return self.otm_put_ivs(S, prices, strikes, dte) + [self.atm_option_iv(S, prices, strikes, dte)] + self.otm_call_ivs(S, prices, strikes, dte)
        return self.otm_put_ivs(S, prices, strikes, dte) + self.otm_call_ivs(S, prices, strikes, dte)
    
    def equal_put_call_skew(self, S: float, call_data: list[list[float, float]], put_data: list[list[float, float]], dte: float):
        """
        parameter
        call_data: a list of lists where each inner list has two elements a a strike and option price
        call_data: a list of lists where each inner list has two elements a a strike and option price
        return
        common_strikes: strikes in our put list and call list that match
        put_ivs: IVs that make the put and call price match, note we use implied rate to get this
        Notes
        To get data use MarketPriceProcessor as such
            call_data = market_processor.all_calls(strike_min, strike_max, exp, steps)
        We still need to add checks to see if the beginning and ending iv are outliers.
        """
        call_data, put_data = np.array(call_data), np.array(put_data)
        common_strikes = np.intersect1d(call_data[:, 0], put_data[:, 0])

        _, call_prices = zip(*call_data[np.isin(call_data[:, 0], common_strikes)])
        _, put_prices = zip(*put_data[np.isin(put_data[:, 0], common_strikes)])

        implied_rates = [implied_rate(call_price, put_price, S, strike, dte) for call_price, put_price, strike in zip(call_prices, put_prices, common_strikes)]

        #call_ivs = [ImpliedVolatility().root_finder(call_price, S, strike, dte, r = rate, otype='call') for call_price, strike, rate in zip(call_prices, common_strikes, implied_rates)]
        #put_ivs = [(strike, ImpliedVolatility().root_finder(put_price, S, strike, dte, r = rate, otype='put')) for put_price, strike, rate in zip(put_prices, common_strikes, implied_rates)]
        strikes_ivs = []
        last_ivs = []
        
        for put_price, strike, rate in zip(put_prices, common_strikes, implied_rates):
            iv = ImpliedVolatility().root_finder(put_price, S, strike, dte, r = rate, otype='put')

            if iv > .05:
                
                strikes_ivs.append((strike, iv))

        cleaned_strikes_ivs = []
        threshold = .07

        for i in range(len(strikes_ivs)):
            strike, iv = strikes_ivs[i]

            if i == 0 or i == len(strikes_ivs) - 1:
                cleaned_strikes_ivs.append((strike, iv))
                continue

            _, iv_prev = strikes_ivs[i - 1]
            _, iv_next = strikes_ivs[i + 1]
            local_avg = (iv_prev + iv_next) / 2

            if (iv - local_avg) > threshold:
                print(f"Removed IV: {iv} at strike {strike} with local avg {local_avg}")
                continue

            cleaned_strikes_ivs.append((strike, iv))

        return cleaned_strikes_ivs

    def otm_put_call_surface(self, S, price_matrix, strike_matrix, dtes):
        return [self.otm_put_call_skew(S, prices, strikes, dte) for prices, strikes, dte in zip(price_matrix, strike_matrix, dtes)]
    
    def get_ivs(self, S: float, prices: list[float], strikes: list[float], dte: float):

        new_ivs = []

        for i in range(len(prices)):
            if S < strikes[i]:
                iv = ImpliedVolatility().root_finder(prices[i], S, strikes[i], dte, otype='call')
                new_ivs.append((strikes[i], iv))
                
            elif S > strikes[i]:
                iv = ImpliedVolatility().root_finder(prices[i], S, strikes[i], dte, otype='put')
                new_ivs.append((strikes[i], iv))

        new_ivs = np.array(new_ivs)
        return new_ivs