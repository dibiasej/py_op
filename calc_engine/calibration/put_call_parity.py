from scipy.optimize import brentq
import numpy as np

def implied_rate(call_price, put_price, S, K, dte, fallback_rate=0.04) -> float:
    """
    Compute implied risk-free rate from put-call parity.
    Falls back to default rate if brentq fails.
    """
    func = lambda r: call_price - put_price - S + K * np.exp(-r * dte)
    
    try:
        if np.sign(func(-1.0)) == np.sign(func(1.0)):
            raise ValueError("No sign change in interval.")
        return brentq(func, -1.0, 1.0)
    except:
        print(f"in fallback")
        return fallback_rate