#from calc_engine.greeks.analytical_greeks import AnalyticalVega
#from calc_engine.option_pricing import analytical_solutions as an
from ..greeks.analytical_greeks import AnalyticalVega
from ..option_pricing import analytical_solutions as an
from calc_engine.misc_funcs.put_call_parity import implied_rate

import numpy as np
from scipy.optimize import root
from scipy.stats import norm
"""
This module is for trying the strategy pattern for iv

In the future we will have to fix our ImpliedVolatility.calculate() method to accept **kwargs so it is more robust.
Note: We want to add sabr normal vol to this
Note: we want to add the numerical bisection method to this
Note: we add sabr normal vol here because sabr is usually used to find IV's not prices, we take the iv we get and plug it into a model like bsm,
    Also the sabr normal vol function we have here needs to be divided by 100 to get the bsm iv
Note: We also want to add dupires formula here for local vol but that depends on finite difference methods for the greeks so we need to build that first
Note: We also want to implement a scipy optimize fsolve method
"""
class ImpliedVolatilityMethod:

    """
    For now we will use a dependency injection method to set our option pricing model, ie we create the pricing factory then create the model then
    pass it into our iv calc method

    Alternatively we can have the pricing factory inside the init of this class
        self.model = an.AnalyticalPriceFactory().create_model(model)
    """

    def __init__(self, model) -> None:
        self.model = model

    def get_model(self):
        return self.model
    
    def set_model(self, new_model):
        self.model = new_model
        
    def calculate(self):
        pass

class NewtonsMethod(ImpliedVolatilityMethod):

    def __init__(self, model = an.BlackScholesMertonAnalytical()) -> None:
        super().__init__(model)

    def calculate(self, market_price: float, S: int, K: int, T: float, r: float = .05, initial_guess: float = .2, otype: str = "call", q = .02, **kwargs) -> float:
    
        xnew: float = initial_guess
        xold: float = 1 - xnew
        for i in range(100):
            if abs(xnew - xold) < .001:
                
                return round(xnew, 3)
            else:
                xold = xnew

                if otype == "call":
                    xnew = xnew - ((self.model.call(S, K, T, xold, r, q, **kwargs) - market_price) / AnalyticalVega.calculate(S, K, T, xold, r, q))

                elif otype == 'put':
                    xnew = xnew - ((self.model.put(S, K, T, xold, r, q, **kwargs) - market_price) / AnalyticalVega.calculate(S, K, T, xold, r, q))
                
        return False
    
class BisectionMethod(ImpliedVolatilityMethod):

    def __init__(self, model = an.BlackScholesMertonAnalytical()) -> None:
        super().__init__(model)

    def calculate(self, market_price: float, S: int, K: int, T: float, r: float = .05, initial_guess: float = .2, lower_vol: float = 0.0, upper_vol: float = 5.0, otype: str = "call", q = .02, **kwargs):

        iv = initial_guess

        for _ in range(1000):

            if otype == 'call':
                model_price = self.model.call(S, K, T, iv, r, q, **kwargs)
            else:
                model_price = self.model.put(S, K, T, iv, r, q, **kwargs)
            
            if abs(market_price - model_price) < .001 or upper_vol - lower_vol < .001:
                return iv

            if market_price - model_price > 0:
                lower_vol = iv
            else:
                upper_vol = iv

            iv = (lower_vol + upper_vol) / 2
            #print(f"iv {iv}")

        return False

class RootFinder(ImpliedVolatilityMethod):

    def __init__(self, model = an.BlackScholesMertonAnalytical()) -> None:
        super().__init__(model)

    def calculate(self, market_price: float, S: float, K: int, T: float, r: float = .05, initial_guess: float = .15, otype: str = "call", q = .02, **kwargs) -> float:

        if otype == "call":
            
            root_fn = lambda x: self.model.call(S, K, T, x, r, q, **kwargs) - market_price

        elif otype == "put":
            
            root_fn = lambda x: self.model.put(S, K, T, x, r, q, **kwargs) - market_price

        return root(root_fn,initial_guess)['x'][0]

class ImpliedVolatility:

    def __init__(self, model = an.BlackScholesMertonAnalytical()) -> None:
        """
        We can pass in any option model into this but we set black scholes analytical as our base case
        Also we might was to add a __call__ method that excepts the bsm parameters and always defaults to newtons method unless we explicitly define the iv calc method we are using
        """
        self.model = model
    
    def newtons_method(self, market_price, S, K, T, r = .05, initial_guess = .2, otype = "call", q = 0, **kwargs):
        params = {'market_price': market_price, 'S': S, 'K': K, 'T': T, 'r': r, 'initial_guess': initial_guess, 'otype': otype, 'q': q, **kwargs}
        return NewtonsMethod(self.model).calculate(**params)
    
    def root_finder(self, market_price, S, K, T, r = .05, initial_guess = .2, otype = "call", q = 0, **kwargs):
        params = {'market_price': market_price, 'S': S, 'K': K, 'T': T, 'r': r, 'initial_guess': initial_guess, 'otype': otype, 'q': q, **kwargs}
        #return RootFinder(self.model).calculate(market_price, S, K, T, r, otype, q)
        return RootFinder(self.model).calculate(**params)
    
    def bisection_method(self, market_price, S, K, T, r = .05, initial_guess = .2, otype = "call", q = 0, **kwargs):
        params = {'market_price': market_price, 'S': S, 'K': K, 'T': T, 'r': r, 'initial_guess': initial_guess, 'otype': otype, 'q': q, **kwargs}
        return BisectionMethod(self.model).calculate(**params)


class SkewCalculator:

    def __init__(self, iv_calculator: ImpliedVolatilityMethod = RootFinder()):
        self.iv_calculator = iv_calculator

    def skew_slope_arbitrage_bound(self, S: float, K: float, dte: float, iv: float, r: float = 0, q: float = 0) -> float:
        d1 = (np.log(S/K) + (r - q + 0.5*iv**2)*dte) / (iv*np.sqrt(dte))
        d2 = d1 - iv*np.sqrt(iv)

        numerator = norm.cdf(d2)
        denominator = K * np.sqrt(dte) * norm.pdf(d2)

        return numerator / denominator

    def calculate_call_skew(self, S: float, call_prices: list[float], strikes: list[float], dte: int, r: float = 0.04, initial_guess: float = 0.15, q: float = 0):

        call_skew_data = [(self.iv_calculator.calculate(price, S, strike, dte, r=r, initial_guess=initial_guess, otype="call", q=q), strike) for price, strike in zip(call_prices, strikes)]
        ivs, strikes = zip(*call_skew_data)
        return ivs, strikes
    
    def calculate_put_skew(self, S: float, put_prices: list[float], strikes: list[float], dte: int, r: float = 0.04, initial_guess: float = 0.15, q: float = 0):

        put_skew_data = [(self.iv_calculator.calculate(price, S, strike, dte, r=r, initial_guess=initial_guess, otype="put", q=q), strike) for price, strike in zip(put_prices, strikes)]
        ivs, strikes = zip(*put_skew_data)
        return ivs, strikes

    def calculate_otm_skew(self, S: float, otm_prices: list[float], strikes: list[float], dte: int, r: float = 0.04, initial_guess: float = 0.15, q: float = 0):
        """
        This method calculates implied vol for otm puts and calls. Get the data for this from the data structure OptionChain.get_otm_skew_prices
        Remember always pass in dte / 365
        Returns:

        Arguments:

        """

        otm_skew_data = [
                (self.iv_calculator.calculate(price, S, strike, dte, r=r, initial_guess=initial_guess, otype="put" if strike < S else "call", q=q), strike)
                    for price, strike in zip(otm_prices, strikes) if strike != S
                ]
        
        ivs, strikes = zip(*otm_skew_data)

        return ivs, strikes

    def calculate_parity_skew1(self, S: float, put_prices: list[float], call_prices: list[float], strikes: list[float], dte: int, r: float = 0.04, initial_guess: float = 0.15, q: float = 0):
        """
        This method calculates implied vol using put call parity to get an implied rate and then we calculate otm put and call iv. Get the data for this from the data structure OptionChain.get_equal_skew_prices
        We only do the iv calculation for puts, but generally I find if we do it for calls no matter what the iv is the same for the puts and the calls
        so to save on computation time we only do puts.
        Remember always pass in dte / 365
        Arguments:

        Returns:
        """

        put_ivs = []

        for put_price, call_price, strike in zip(put_prices, call_prices, strikes):

            i_rate = implied_rate(call_price, put_price, S, strike, dte)
            put_iv = self.iv_calculator.calculate(put_price, S, strike, dte, r=i_rate, initial_guess=initial_guess, otype = "put", q=q)

            if put_iv > 0.067:
                put_ivs.append((put_iv, strike))

        put_ivs_cleaned = []
        threshold = 0.07

        for i in range(len(put_ivs)):
            put_iv, strike = put_ivs[i]

            if i == 0 or i == len(put_ivs) - 1:
                put_ivs_cleaned.append((put_iv, strike))
                continue

            put_iv_prev, _ = put_ivs[i - 1]
            put_iv_next, _ = put_ivs[i + 1]
            put_local_avg = (put_iv_prev + put_iv_next) / 2

            if (put_iv - put_local_avg) > threshold:
                continue

            put_ivs_cleaned.append((put_iv, strike))

        return put_ivs_cleaned
    
    def calculate_parity_skew(self, S: float, put_prices: list[float], call_prices: list[float], strikes: list[float], dte: int, r: float = 0.04, initial_guess: float = 0.15, q: float = 0):
        """
        This method calculates implied vol using put call parity to get an implied rate and then we calculate otm put and call iv. Get the data for this from the data structure OptionChain.get_equal_skew_prices
        We only do the iv calculation for puts, but generally I find if we do it for calls no matter what the iv is the same for the puts and the calls
        so to save on computation time we only do puts.
        Remember always pass in dte / 365
        Arguments:

        Returns:
        """

        put_ivs = []

        for put_price, call_price, strike in zip(put_prices, call_prices, strikes):

            i_rate = implied_rate(call_price, put_price, S, strike, dte)
            put_iv = self.iv_calculator.calculate(put_price, S, strike, dte, r=i_rate, initial_guess=initial_guess, otype = "put", q=q)

            if put_iv > 0.067:
                put_ivs.append((put_iv, strike))
        
        put_ivs_cleaned = []

        T = dte
        #log_moneyness = np.log(strikes / S)

        for i in range(1, len(put_ivs) - 1):

            iv, K = put_ivs[i]
            iv_prev, _ = put_ivs[i - 1]
            iv_next, _ = put_ivs[i + 1]

            # Convert to total variance
            w = iv**2 * T
            w_prev = iv_prev**2 * T
            w_next = iv_next**2 * T

            median = np.median([w_prev, w, w_next])

            # Adaptive threshold (15% of local level)
            if abs(w - median) > 0.15 * median:
                continue

            put_ivs_cleaned.append((iv, K))

        return put_ivs_cleaned
    
    def calculate_parity_skew_all_data(self, S: float, put_prices: list[float], call_prices: list[float], strikes: list[float], dte: int, r: float = 0.04, initial_guess: float = 0.15, q: float = 0):
        """
        This method calculates implied vol using put call parity to get an implied rate and then we calculate otm put and call iv. Get the data for this from the data structure OptionChain.get_equal_skew_prices
        We only do the iv calculation for puts, but generally I find if we do it for calls no matter what the iv is the same for the puts and the calls
        so to save on computation time we only do puts.
        Remember always pass in dte / 365
        Arguments:

        Returns:
        """

        put_ivs = []

        for put_price, call_price, strike in zip(put_prices, call_prices, strikes):

            i_rate = implied_rate(call_price, put_price, S, strike, dte)
            put_iv = self.iv_calculator.calculate(put_price, S, strike, dte, r=i_rate, initial_guess=initial_guess, otype = "put", q=q)

            if put_iv > 0.067:
                put_ivs.append((put_iv, strike))

        return put_ivs

class TermStructureCalculator:

    def __init__(self, iv_calculator: ImpliedVolatilityMethod = RootFinder()):
        self.iv_calculator = iv_calculator

    def calculate_term_structure_call(self, S: float, strike: float, call_prices: list[float], dtes: list[float], r: float = 0.04, q: float = 0) -> tuple[list[float], list[float]]:
        dtes = np.array(dtes)

        call_ivs = [self.iv_calculator.calculate(call_price, S, strike, t, r=r, initial_guess=.15, otype = "call", q=q) for call_price, t in zip(call_prices, dtes)]

        return call_ivs, dtes
    
    def calculate_term_structure_put(self, S: float, strike: float, put_prices: list[float], dtes: list[float], r: float = 0.04, q: float = 0) -> tuple[list[float], list[float]]:
        dtes = np.array(dtes)

        put_ivs = [self.iv_calculator.calculate(put_price, S, strike, t, r=r, initial_guess=.15, otype = "put", q=q) for put_price, t in zip(put_prices, dtes)]

        return put_ivs, dtes
    
    def calculate_atf_term_structure(self, S: float, forward_strikes: list[float], call_prices: list[float], put_prices: list[float], dtes: list[float], r: float = 0.04, q: float = 0) -> tuple[list[float], list[float]]:
        """
        This method calculates the Implied Volatility term structure for at the forward options.
        Important:
        - You must pass in a list of at the forward strikes across expiries
        - Call and put prices must also be at the forward.
        - The best place to get the data for this method is OptionChain.get_equal_term_structure_atf_prices().
        - Make sure the dtes you pass in is divided by 365.
        """
        i_rates = [implied_rate(call_price, put_price, S, f_strike, dte) for call_price, put_price, f_strike, dte in zip(call_prices, put_prices, forward_strikes, dtes)]

        call_ivs = [self.iv_calculator.calculate(call_price, S, f_strike, t, r=i_rate, initial_guess=.15, otype = "call", q=q) for call_price, f_strike, t, i_rate in zip(call_prices, forward_strikes, dtes, i_rates)]
        put_ivs = [self.iv_calculator.calculate(put_price, S, f_strike, t, r=i_rate, initial_guess=.15, otype = "put", q=q) for put_price, f_strike, t, i_rate in zip(put_prices, forward_strikes, dtes, i_rates)]

        return put_ivs, call_ivs, i_rates

def main():

    """
    Below we show an ex of how we currently can use this

    Note: Alternatively we an set the AnalyticalPriceFactory() inside the ImpliedVolatilityMethod class constructor so we only have to pass in
    a str to NewtonsMethod but I don't think this is very robust 
    """
    pricing_factory = an.AnalyticalPriceFactory()

    bsm = pricing_factory.create_model("bsm")

    iv = NewtonsMethod(bsm)

    return None

if __name__ == "__main__":
    print(main())