#from calc_engine.greeks.analytical_greeks import AnalyticalVega
#from calc_engine.option_pricing import analytical_solutions as an
from ..greeks.analytical_greeks import AnalyticalVega
from ..option_pricing import analytical_solutions as an

import numpy as np
from scipy.optimize import root
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

        if T >= 1:
            T /= 365

        elif T == 0:
            T = .5 / 365
    
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