import numpy as np

def variance_swap_approximation(S, put_prices, call_prices, strikes, dte, r):
    """
    This function should use data from option_chain.get_equal_skew_prices()
    Important do not pass in a dte/365
    To make this a true vix calculation we would need to calculate this for two dtes and do a linear interpolation
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

def skew_swap_approximation(S, put_prices, call_prices, strikes, dte):

    strikes = np.asarray(strikes, dtype=float)
    put_prices = np.asarray(put_prices, dtype=float)
    call_prices = np.asarray(call_prices, dtype=float)

    # Sort everything by strike just in case
    idx = np.argsort(strikes)
    strikes = strikes[idx]
    put_prices = put_prices[idx]
    call_prices = call_prices[idx]

    T = dte / 365.0
    ks_sum = 0.0

    for i in range(1, len(strikes) - 1):
        K = strikes[i]

        # Central-difference strike spacing
        dK = (strikes[i + 1] - strikes[i - 1]) / 2.0

        # Equation (6): use puts for K < S0, calls for K > S0
        # At K = S0, integrand is zero anyway since (S0 - K) = 0
        if K < S:
            Q = put_prices[i]
        elif K > S:
            Q = call_prices[i]
        else:
            Q = 0.0

        integrand = Q * (S - K) / (K ** 2)
        ks_sum += integrand * dK

    Ks = (2.0 / (T * S)) * ks_sum
    return Ks

def linear_interpolated_iv_v1(iv1, iv2, dte1, dte2, target_date):
    """
    This is the pure math calculation for interpolated iv.
    This will go in calc_engine
    """
    iv = iv1 + ((np.log(target_date) - np.log(dte1)) / (np.log(dte2) - np.log(dte1))) * (iv2 - iv1)
    return iv

def forward_volatility(iv1, iv2, dte1, dte2) -> float:
    """
    This function calculates the forward volatility, ie the volatility expected between two dated T1 and T2.
    Arguments:
    iv1 float: Can be atm iv, var swap, any vol at earlier date
    iv2 float: Can be atm iv, var swap, any vol at later date
    dte1 (int): This is the earlier days to expiration, this should NOT be divided by 365
    dte2 (int): This is the later days to expiration, this should NOT be divided by 365
    """
    T1 = dte1/365
    T2 = dte2/365
    return np.sqrt((T2*iv2**2 - T1*iv1**2) / (T2 - T1))