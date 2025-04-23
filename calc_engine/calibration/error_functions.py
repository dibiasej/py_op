import numpy as np
from scipy.optimize import minimize

# two main types of option optimization error functions are 1) based on price 2) based on IVs
""" Lets try to change this so for every objective function we pass in a numpy array of strikes and prices into sum of squres so it calculates 
the whole sum of squred errors of skew in a fact vectorized manner
"""

def sum_of_squares_clean(S: float, strikes: np.ndarray, prices: np.ndarray, T: float, model, otype: str = 'call', **kwargs) -> float:
    """
    Params:
    This is the latest sum of squares I made for the r Bergomi model eventually I want to delete the other and have this only
    """

    if otype == 'call':
        model_call = model.call(S, T, strikes, **kwargs)
        sum_ = np.sum((prices - model_call)**2)
    else:
        model_put = model.put(S, T, strikes, **kwargs)
        sum_ = np.sum((prices - model_put)**2)

    return sum_

def sum_of_squares(S: float, strikes: list[int], prices: list[float], T: float, model, otype = 'call', **kwargs):
    """
    Params:
    float S: Spot price of asset.
    list[int] strikes: List of strike prices, we might turn this into an np.array.
    float T: time to expiration, should be of form days/252.
    list[int] prices: Market option prices, we need might turn this into a np.array.
    model: Our specific option pricing model, must have a call or put method.

    Notes: 
    The lenght of strikes and prices must be equal
    """
    sum_ = 0

    if otype == 'call':

        for i in range(len(prices)): # we should change this so there is no loop and we work with numpy arrays
            
            model_call = model.call(S, strikes[i], T, **kwargs)
            if isinstance(model_call, list):
                sum_ += (prices[i] - model_call[i])**2

            else:
                sum_ += (prices[i] - model_call)**2

    else:

        for i in range(len(prices)):
            model_put = model.put(S, strikes[i], T, **kwargs)
            if isinstance(model_put, list):
                sum_ += (prices[i] - model_put[i])**2
            
            else:
                sum_ += (prices[i] - model_put)**2

    return sum_

def sum_of_squares_puts_and_calls(S: float, strikes: np.ndarray, market_prices: np.ndarray, T: float, model, **kwargs) -> float:
    """
    Params:
    float S: Spot price of asset.
    list[int] strikes: List of strike prices, we might turn this into an np.array.
    float T: time to expiration, should be of form days/252.
    list[int] prices: Market option prices, we need might turn this into a np.array.
    model: Our specific option pricing model, must have a call or put method.

    Notes: 
    This error function is used when our price data is a list composed of both otm calls and puts.
    """
    assert len(strikes) == len(market_prices), "Strikes and market prices must be the same length"

    put_indices = np.where(strikes <= S)[0]
    call_indices = np.where(strikes > S)[0]

    market_call = market_prices[call_indices]
    market_put = market_prices[put_indices]

    call_strikes = strikes[call_indices]
    put_strikes = strikes[put_indices]

    model_call = model.call(S, T, call_strikes, **kwargs)
    sum_call = np.sum((market_call - model_call)**2)
    
    model_put = model.put(S, T, put_strikes, **kwargs)
    sum_put = np.sum((market_put - model_put)**2)

    return sum_call + sum_put

def sum_of_squares_heston_surface(S: float, strikes: list[int], prices: list[list[float]], dtes: list[float], model, otype = 'call', **kwargs):
    """
    Params:
    float S: Spot price of asset.
    list[int] strikes: List of strike prices, we might turn this into an np.array.
    float T: time to expiration, should be of form days/252.
    list[int] prices: Market option prices, we need might turn this into a np.array.
    model: Our specific option pricing model, must have a call or put method.

    Notes: 
    The lenght of strikes and prices must be equal
    """
    sum_ = 0

    if otype == 'call':
        for i, dte in enumerate(dtes):

            for j, strike in enumerate(strikes):

                model_call = model.call(S, strike, dte, **kwargs)

                if isinstance(model_call, list):
                    # Note: I dont think this portion is necessary but we will refrain from omitting it right now
                    sum_ += (prices[i] - model_call[i])**2

                else:
                    sum_ += (prices[i][j] - model_call)**2

    else:
        for i, dte in enumerate(dtes):
            for j, strike in enumerate(strikes):

                model_put = model.put(S, strike, dte, **kwargs)
                if isinstance(model_put, list):
                    sum_ += (prices[i] - model_put[i])**2
                
                else:
                    sum_ += (prices[i][j] - model_put)**2
    return sum_

def sum_of_squares_heston_skew(S: float, strikes: list[int], prices: list[float], dte: float, model, otype = 'call', **kwargs):

    if dte > 1:
        dte = dte/252

    sum_ = 0

    for i, strike in enumerate(strikes):

        if otype == 'call':

            model_price = model.call(S, strike, dte, **kwargs)
        
        else:
            model_price = model.put(S, strike, dte, **kwargs)

        sum_ += (prices[i] - model_price)**2

    return sum_

def sse_heston_fft(S: float, strikes: list[float], prices: list[list[float]], dtes: list[float], model, otype: str = 'call', **kwargs):
    """
    This error function is specifically for the fft method with the heston
    """
    if dtes[1] >= 1:
        dtes = np.array(dtes) / 252
    sum_ = 0
    for i, price_list in enumerate(prices):
        if otype == 'call':
            model_prices = model.call(S, strikes, dtes[i], **kwargs)

        else:
            model_prices = model.put(S, strikes, dtes[i], **kwargs)

        #print(f"dte {dtes[i]}")
        #print(f"Model Prices {model_prices}")
        #print(f"Observed Prices: {price_list}\n")


        sum_ += sum((np.array(price_list) - np.array(model_prices)) ** 2)

    return sum_


def relative_sum_of_squares(S: float, strikes: list[int], prices: list[float], T: float, model, otype = 'call', **kwargs):
    """
    Params:
    float S: Spot price of asset.
    list[int] strikes: List of strike prices, we might turn this into an np.array.
    float T: time to expiration, should be of form days/252.
    list[int] prices: Market option prices, we need might turn this into a np.array.
    model: Our specific option pricing model, must have a call or put method.
    """
    sum_ = 0

    if otype == 'call':
        for i in range(len(prices)):
            sum_ += ((prices[i] - model.call(S, strikes[i], T, **kwargs))**2) / prices[i]

    else:
        for i in range(len(prices)):
            sum_ += ((prices[i] - model.put(S, strikes[i], T, **kwargs))**2 ) / prices[i]

    return sum_

def sse_log_ivs(S: int, strikes: list[float], ivs: list[float], T: float, model, **kwargs):
    """
    Generally we use this for the SABR model but eventually we want to modify this so it can work with other models
    For this error function we dont need to worry about a parameter for calls and puts because we are passing in ivs not prices
    """
    sum_ = 0
    for i in range(len(ivs)):
        if S == strikes[i]:
            sig_atm = (1/2)*model.lognormal_vol(S, strikes[i - 1], T, **kwargs) + (1/2)*model.lognormal_vol(S, strikes[i + 1], T, **kwargs)
            sum_=(ivs[i] - sig_atm)**2
            continue

        sig_b = model.lognormal_vol(S, strikes[i], T, **kwargs)
        sum_ += (ivs[i] - sig_b)**2

    return sum_