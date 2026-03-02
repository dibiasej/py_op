import numpy as np

def variance_swap_approximation(S, put_prices, call_prices, strikes, dte, r):
    """
    This function should use data from option_chain.get_equal_skew_prices()
    Important do not pass in a dte/365
    """
    strikes = np.array(strikes, dtype=float)
    T = dte/365
    F = S*np.exp(r*T)
    
    moneyness_arr = strikes / F
    atm_idx = np.abs(moneyness_arr - 1).argmin()

    x0 = strikes[atm_idx]
    c0 = call_prices[atm_idx]
    p0 = put_prices[atm_idx]
    F0 = x0 + np.exp(r * T) * (c0 - p0)

    K0 = strikes[strikes <= F0].max()
    sigma_2 = 0.0

    for i in range(1, len(strikes) - 1):
        # ΔK_i = (K_{i+1} - K_{i-1}) / 2
        delKi = (strikes[i + 1] - strikes[i - 1]) / 2

        # Q(K_i): OTM put if K < K0, OTM call if K > K0,
        # and for K == K0 use average of put/call (ATM handling)
        if strikes[i] < K0:
            Qi = put_prices[i]
        elif strikes[i] > K0:
            Qi = call_prices[i]
        else:
            Qi = 0.5 * (put_prices[i] + call_prices[i])

        sigma_2 += (delKi * np.exp(r * T) * Qi) / (strikes[i] ** 2)

    # VIX variance
    sigma_2_new = (2 / T) * sigma_2 - (1 / T) * ((F0 / K0) - 1) ** 2
    vix = np.sqrt(sigma_2_new)
    return vix