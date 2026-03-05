import numpy as np

def implied_rate(call_price, put_price, S, K, dte):
    """
    This function is a the solution for the rate in put call parity C - P = S - Ke^(-rT)
    Important: the dte we pass in must be divided by 365, if it is not we will get an incorrect calculation.
    """
    return -np.log((S - call_price + put_price)/K) / dte