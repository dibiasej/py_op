import numpy as np

def variance_swap_fixed_leg(S, put_prices, call_prices, strikes, dte, r):
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

def variance_swap_fixed_leg_neuberger(S, put_prices, call_prices, strikes, dte, r = 0.04, q = 0):
    """
    This is the variance swap contract from Neurbergers paper eq (23)
    """

    strikes = np.asarray(strikes, dtype=float)
    put_prices = np.asarray(put_prices, dtype=float)
    call_prices = np.asarray(call_prices, dtype=float)

    # Sort everything by strike just in case
    idx = np.argsort(strikes)
    strikes = strikes[idx]
    put_prices = put_prices[idx]
    call_prices = call_prices[idx]

    T = dte / 365.0
    put_sum = 0
    call_sum = 0

    F = S * np.exp((r - q) * T)

    for i in range(1, len(strikes) - 1):
        K = strikes[i]

        # Central-difference strike spacing
        dK = (strikes[i + 1] - strikes[i - 1]) / 2.0

        if K < F:
            Q = put_prices[i] / (np.exp(-r*T) * K ** 2)
            put_sum += Q * dK
        elif K > F:
            Q = call_prices[i] / (np.exp(-r*T) * K ** 2)
            call_sum += Q * dK
        else:
            Q = 0.0

    var_swap = 2*put_sum + 2*call_sum
    return var_swap

def entropy_contract_approximation(S, put_prices, call_prices, strikes, dte, r = 0.04):
    """
    This is the entropy contract from Neurbergers skew risk premium paper eq (24)
    """

    strikes = np.asarray(strikes, dtype=float)
    put_prices = np.asarray(put_prices, dtype=float)
    call_prices = np.asarray(call_prices, dtype=float)

    # Sort everything by strike just in case
    idx = np.argsort(strikes)
    strikes = strikes[idx]
    put_prices = put_prices[idx]
    call_prices = call_prices[idx]

    T = dte / 365.0
    put_sum = 0
    call_sum = 0

    F = S * np.exp(r*T)

    for i in range(1, len(strikes) - 1):
        K = strikes[i]

        dK = (strikes[i + 1] - strikes[i - 1]) / 2.0

        if K < F:
            Q = put_prices[i] / (np.exp(-r*T) * K * F)
            put_sum += Q * dK
        elif K > F:
            Q = call_prices[i] / (np.exp(-r*T) * K * F)
            call_sum += Q * dK
        else:
            Q = 0.0

    entropy_contract = 2*put_sum + 2*call_sum
    return entropy_contract

def skew_swap_fixed_leg_peter_lee(S, put_prices, call_prices, strikes, dte):
    """
    This function calculates the skew swap strike given by (6) in Peter Lee's paper
    """

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

def skew_swap_fixed_leg_neuberger(S, put_prices, call_prices, strikes, dte, r = 0.04):
    """
    This function calculates the skew swap strike given by (13) in Neurberger paper
    """

    strikes = np.asarray(strikes, dtype=float)
    put_prices = np.asarray(put_prices, dtype=float)
    call_prices = np.asarray(call_prices, dtype=float)

    # Sort everything by strike just in case
    idx = np.argsort(strikes)
    strikes = strikes[idx]
    put_prices = put_prices[idx]
    call_prices = call_prices[idx]

    T = dte / 365.0
    F = S * np.exp(r*T)

    put_side_sum = 0
    call_side_sum = 0

    for i in range(1, len(strikes) - 1):
        K = strikes[i]

        # Central-difference strike spacing
        dK = (strikes[i + 1] - strikes[i - 1]) / 2.0

        # Equation (6): use puts for K < S0, calls for K > S0
        # At K = S0, integrand is zero anyway since (S0 - K) = 0
        if K < S:
            Q = put_prices[i] * (F - K) / (F * K ** 2) 
            put_side_sum += Q * dK
        elif K > S:
            Q = call_prices[i] * (K - F) / (F * K ** 2)
            call_side_sum += Q * dK
        else:
            Q = 0.0

    Ks = (6.0 / np.exp(-r*T)) * (call_side_sum - put_side_sum)
    return Ks

def model_free_implied_skewness(S, put_prices, call_prices, strikes, dte, r = 0.04, q: float = 0):
    """
    This function calculates the implied skewness as outlines in Variance Risk Premium, Skewness Risk Premium and Equity Expected Returns by Akio Ito
    This is very similar to the other skew swap strikes
    """

    strikes = np.asarray(strikes, dtype=float)
    put_prices = np.asarray(put_prices, dtype=float)
    call_prices = np.asarray(call_prices, dtype=float)

    # Sort everything by strike just in case
    idx = np.argsort(strikes)
    strikes = strikes[idx]
    put_prices = put_prices[idx]
    call_prices = call_prices[idx]

    T = dte / 365.0
    F = S * np.exp((r - q)*T)

    P1 = 0
    P2 = 0
    P3 = 0

    for i in range(1, len(strikes) - 1):
        K = strikes[i]

        # Central-difference strike spacing
        dK = (strikes[i + 1] - strikes[i - 1]) / 2.0

        if K < F:

            Q = put_prices[i]
            P1 += (1 / K**2) * Q * dK
            P2 += (2 / K**2) * (1 - np.log(K/F)) * Q * dK
            P3 += (3 / K**2) * (2*np.log(K/F) - np.log(K/F)**2) * Q * dK

        elif K > F:

            Q = call_prices[i]
            P1 += (1 / K**2) * Q * dK
            P2 += (2 / K**2) * (1 - np.log(K/F)) * Q * dK
            P3 += (3 / K**2) * (2*np.log(K/F) - np.log(K/F)**2) * Q * dK

        else:
            Q = 0.0

    # discount
    P1 = np.exp(r*T)*(-P1)
    P2 = np.exp(r*T)*P2
    P3 = np.exp(r*T)*P3

    mfis = (P3 - 3*P1*P2 + 2*P1**3) / (P2 - P1**2)**(3/2)
    return mfis

def model_free_implied_kurtosis(S, put_prices, call_prices, strikes, dte, r = 0.04, q: float = 0):
    """
    This function is similar to the model free implied skewness from Ito (2025), this is derived using otm puts and calls and is an approximation of the risk-neutral skewness.
    This is from Bakshi, Kapadia and Madan (2003)
    """
    strikes = np.asarray(strikes, dtype=float)
    put_prices = np.asarray(put_prices, dtype=float)
    call_prices = np.asarray(call_prices, dtype=float)

    # Sort everything by strike just in case
    idx = np.argsort(strikes)
    strikes = strikes[idx]
    put_prices = put_prices[idx]
    call_prices = call_prices[idx]

    T = dte / 365.0
    F = S * np.exp((r - q)*T)

    V = 0
    W = 0
    X = 0

    for i in range(1, len(strikes) - 1):
        K = strikes[i]

        # Central-difference strike spacing
        dK = (strikes[i + 1] - strikes[i - 1]) / 2.0

        if K < F:

            P = put_prices[i]
            V += ((2*(1 + np.log(F / K))) / K**2) * P * dK
            W += (-(6*np.log(F / K) + 3*np.log(F / K)**2) / K**2) * P * dK
            X += ((12*np.log(F / K)**2 + 4*np.log(F / K)**3) / K**2) * P * dK


        elif K > F:

            C = call_prices[i]
            V += ((2*(1 - np.log(K / F))) / K**2) * C * dK
            W += ((6*np.log(K / F) - 3*np.log(K / F)**2) / K**2) * C * dK
            X += ((12*np.log(K / F)**2 - 4*np.log(K / F)**3) / K**2) * C * dK


        else:
            Q = 0.0

    mu = np.exp(r*T) - 1 - 0.5*np.exp(r*T)*V - (1/6)*np.exp(r*T)*W - (1/24)*np.exp(r*T)*X
    KURT = (np.exp(r*T)*X - 4*mu*np.exp(r*T)*W + 6*np.exp(r*T)*mu**2*V - 3*mu**4) / (np.exp(r*T)*V - mu**2)**2
    return KURT

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

def atm_skew_approx_two_factor_smile_3_bergomi(T: list[float], nu: float, theta: float, k1: float, k2: float, rho_x1x2: float, rho_sx1: float, rho_sx2: float):
    """
    This is the atmf skew term structure approximation used in Bergomis 2 factor model from smile dynamics 3. It approximates 95% - 105% atmf skew
    It is a very good approximation but we need to calibrate the parameters first but below I give got starting approximations
        nu=1.3, theta=0.28, k1=8, k2=0.35, rho_x1x2=0, rho_sx1=-0.7, rho_sx2=-0.357
    """
    alpha = 1 / np.sqrt((1 - theta)**2 + theta**2 + 2 * rho_x1x2 * theta * (1 - theta))

    term1 = (1 - theta) * rho_sx1 * (k1 * T - (1 - np.exp(-k1 * T))) / (k1**2 * T**2)
    term2 = theta * rho_sx2 * (k2 * T - (1 - np.exp(-k2 * T))) / (k2**2 * T**2)

    return nu * alpha * (term1 + term2) * -np.log(1.05 / 0.95)

def vol_of_vol_two_factor_smile_3_bergomi(T: list[float], nu: float, theta: float, k1: float, k2: float, rho_x1x2: float, zeta: float=1.0):
    """
    Instantaneous volatility of variance swap volatility sqrt(V_0,T).

    This is the vol-of-vol term structure. Ie we can plug in multiple T's and easily get a term structure
    """
    alpha = 1 / np.sqrt((1 - theta)**2 + theta**2+ 2 * rho_x1x2 * theta * (1 - theta))

    a1 = (1 - theta) * (1 - np.exp(-k1 * T)) / (k1 * T)
    a2 = theta * (1 - np.exp(-k2 * T)) / (k2 * T)

    vov = nu * zeta * alpha * np.sqrt(a1**2 + a2**2 + 2 * rho_x1x2 * a1 * a2)
    return vov