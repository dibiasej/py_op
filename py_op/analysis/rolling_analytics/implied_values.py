import numpy as np

from py_op.calc_engine.vol_engine.iv_calc import RootFinder
from py_op.calc_engine.vol_engine.vol_funcs import linear_interpolated_iv_v1
from py_op.calc_engine.misc_funcs.put_call_parity import implied_rate
from py_op.data.builders.option_chain_builder import create_chain_series
from py_op.utils.date_utils import find_bracketing_dtes

"""
This is used to calculate all implied values 
EX:
 -- Interpolated constant maturity IV
 -- Implied skew 
 -- implied parameters from models like SABR, Heston, rBergomi
"""

def implied_skew_moneyness(ticker: str, start_date: str, end_date: str, target_dte: int, moneyness: float=0.1, steps: int=1, max_days_diff: int=20):
    """
    This is the version of implied skew from Euan Sinclair retail option trading. It is also number 6 in the flash cards ipynb
    The issue with this is we need to make it constant maturity right now it just fetches the maturity closest to dte
    Note: (2/16/26) we will change this so it gets a constant maturity dte not the closest
    """
    chain_series = create_chain_series(ticker, start_date, end_date, steps=steps)
    implied_skews, dates = [], []
    rf = RootFinder()

    for chain in chain_series:
        S = chain.S

        otm_prices, strikes, actual_dte = chain.get_otm_skew_prices(
            dte=target_dte, max_days_diff=max_days_diff
        )

        strikes = np.array(strikes, dtype=float)
        m_arr = strikes / S

        idx_put  = np.abs(m_arr - (1 - moneyness)).argmin()
        idx_call = np.abs(m_arr - (1 + moneyness)).argmin()

        put_price,  K_put  = float(otm_prices[idx_put]),  float(strikes[idx_put])
        call_price, K_call = float(otm_prices[idx_call]), float(strikes[idx_call])

        T = actual_dte / 365.0  # use the returned DTE for pricing

        put_iv  = rf.calculate(put_price,  S, K_put,  T, otype="put")
        call_iv = rf.calculate(call_price, S, K_call, T, otype="call")

        denom = (K_put - K_call) / S
        implied_skews.append((put_iv - call_iv) / denom)
        dates.append(chain.close_date)

    return implied_skews, dates

def implied_fixed_strike_skew(ticker: str, start_date: str, end_date: str, target_dte: int, moneyness: float = .1, steps: int = 1, max_days_diff: int = 10):
    """
    This is the version of implied skew from Collin Bennets book Trading Volatility
    90% - 110% or 90% - 100%
    """
    chain_series = create_chain_series(ticker, start_date, end_date, steps=steps)
    implied_skews, dates = [], []
    rf = RootFinder()

    for chain in chain_series:
        S = chain.S

        otm_prices, strikes, actual_dte = chain.get_otm_skew_prices(
            dte=target_dte, max_days_diff=max_days_diff
        )

        strikes = np.array(strikes, dtype=float)
        m_arr = strikes / S

        idx_put  = np.abs(m_arr - (1 - moneyness)).argmin()
        idx_call = np.abs(m_arr - (1 + moneyness)).argmin()

        put_price,  K_put  = float(otm_prices[idx_put]),  float(strikes[idx_put])
        call_price, K_call = float(otm_prices[idx_call]), float(strikes[idx_call])

        T = actual_dte / 365.0  # use the returned DTE for pricing

        put_iv  = rf.calculate(put_price,  S, K_put,  T, otype="put")
        call_iv = rf.calculate(call_price, S, K_call, T, otype="call")

        implied_skews.append(put_iv - call_iv)
        dates.append(chain.close_date)

    return implied_skews, dates

def constant_maturity_atm_iv(ticker: str, start_date: str, end_date: str, target_dte: int, q: float = 0.0, steps: int = 1):
    """
    This function uses linear interpolation to get a atm constant matuity iv.
    Right now it only gets put iv, put it uses an implied rate so they should be equal, but in the future we will test for call iv.
    We will make two other functions one that gets atm put iv and one that gets atm call iv.
    We should add max_days_diff = 0
    """
    chain_series = create_chain_series(ticker, start_date, end_date, steps=steps)

    ivs, dates = [], []
    t_target = target_dte / 365

    for chain in chain_series:
        dte_list = chain.get_common_dtes()
        dte_lo, dte_hi = find_bracketing_dtes(dte_list, target_dte)

        S = chain.S

        # guard: if chain has no usable DTEs
        if dte_lo is None and dte_hi is None:
            continue

        # exact match or clamped
        if dte_hi is None:

            call = chain.get_option(S, otype="call", dte=dte_lo, max_days_diff = 0)
            put  = chain.get_option(S, otype="put",  dte=dte_lo, max_days_diff = 0)

            T = put.dte / 365  # use the contract's DTE to be consistent with your objects
            i_rate = implied_rate(call.price, put.price, S, put.strike, T)

            iv = RootFinder().calculate(put.price, S, put.strike, T, r=i_rate, otype="put", q=q)

        # interpolate between two expiries
        else:

            call_lo = chain.get_option(S, otype="call", dte=dte_lo, max_days_diff = 0)
            put_lo = chain.get_option(S, otype="put", dte=dte_lo, max_days_diff = 0)

            call_hi = chain.get_option(S, otype="call", dte=dte_hi, max_days_diff = 0)
            put_hi = chain.get_option(S, otype="put", dte=dte_hi, max_days_diff = 0)

            T1 = put_lo.dte / 365
            T2 = put_hi.dte / 365
            i_rate1 = implied_rate(call_lo.price, put_lo.price, S, put_lo.strike, T1)
            i_rate2 = implied_rate(call_hi.price, put_hi.price, S, put_hi.strike, T2)

            iv1 = RootFinder().calculate(put_lo.price, S, put_lo.strike, T1, r=i_rate1, otype="put", q=q)
            iv2 = RootFinder().calculate(put_hi.price, S, put_hi.strike, T2, r=i_rate2, otype="put", q=q)

            t1 = dte_lo / 365
            t2 = dte_hi / 365
            iv = linear_interpolated_iv_v1(iv1, iv2, t1, t2, t_target)

        ivs.append(iv)
        dates.append(chain.close_date)

    return ivs, dates

class RollingTermStructure:
    """
    This class will be used to get rolling term structure analytics
    First we will make it compatible with the z-score analytics from Euan sinclairs book
    Also make it compatible with rolling dte2 - dte1 where dte2 > dte1
        Either add argument to each method that says z_score = True, or make completely seperate methods eg, atmf_zscore
    We may make something like this called RollingSkew as well   
    """

    def atmf():
        pass

    def constant_maturity():
        pass

    def variance_swap():
        pass


class RollingSkew:
    pass

"""
I think I want to make classes for Skew and TermStructure that have methods for calculating data, then methods for getting certain values
like fixed skew, moneyness skew, delta skew
"""

"""
Functions I might get rid of
"""

def constant_maturity_implied_skew_moneyness(ticker: str, start_date: str, end_date: str, target_dte: int, moneyness: float = .1, steps: int = 1):
    """
    This gets the constant maturity 90% - 110% skew, the value is very similar to the implied_skew_moneyness so I might get rid of it
    """
    chain_series = create_chain_series(ticker, start_date, end_date, steps=steps)
    rf = RootFinder()
    implied_skews, dates = [], []

    for chain in chain_series:
        S = chain.S
        dte_list = chain.get_common_dtes()
        dte_lo, dte_hi = find_bracketing_dtes(dte_list, target_dte)

        if dte_lo is None and dte_hi is None:
            continue

        if dte_hi is None:

            otm_prices, strikes, actual_dte = chain.get_otm_skew_prices(dte=dte_lo, max_days_diff=0)    
            strikes = np.array(strikes, dtype=float)
            moneyness_arr = strikes/S

            idx_put  = np.abs(moneyness_arr - (1 - moneyness)).argmin()
            idx_call = np.abs(moneyness_arr - (1 + moneyness)).argmin()

            put_price,  K_put  = float(otm_prices[idx_put]),  float(strikes[idx_put])
            call_price, K_call = float(otm_prices[idx_call]), float(strikes[idx_call])

            T = actual_dte/365

            put_iv  = rf.calculate(put_price,  S, K_put,  T, otype="put")
            call_iv = rf.calculate(call_price, S, K_call, T, otype="call")

            denom = (K_put - K_call) / S
            implied_skews.append((put_iv - call_iv) / denom)
            dates.append(chain.close_date)

        else:
            otm_prices_lo, strikes_lo, actual_dte_lo = chain.get_otm_skew_prices(dte=dte_lo, max_days_diff=0)
            otm_prices_hi, strikes_hi, actual_dte_hi = chain.get_otm_skew_prices(dte=dte_hi, max_days_diff=0)
            strikes_lo = np.array(strikes_lo, dtype=float)
            strikes_hi = np.array(strikes_hi, dtype=float)

            moneyness_arr_lo = strikes_lo/S
            moneyness_arr_hi = strikes_hi/S

            idx_put_lo  = np.abs(moneyness_arr_lo - (1 - moneyness)).argmin()
            idx_call_lo = np.abs(moneyness_arr_lo - (1 + moneyness)).argmin()

            idx_put_hi  = np.abs(moneyness_arr_hi - (1 - moneyness)).argmin()
            idx_call_hi = np.abs(moneyness_arr_hi - (1 + moneyness)).argmin()

            put_price_lo,  K_put_lo  = float(otm_prices_lo[idx_put_lo]),  float(strikes_lo[idx_put_lo])
            call_price_lo, K_call_lo = float(otm_prices_lo[idx_call_lo]), float(strikes_lo[idx_call_lo])

            put_price_hi,  K_put_hi  = float(otm_prices_hi[idx_put_hi]),  float(strikes_hi[idx_put_hi])
            call_price_hi, K_call_hi = float(otm_prices_hi[idx_call_hi]), float(strikes_hi[idx_call_hi])

            T_lo = actual_dte_lo/365
            T_hi = actual_dte_hi/365

            put_iv_lo  = rf.calculate(put_price_lo,  S, K_put_lo,  T_lo, otype="put")
            call_iv_lo = rf.calculate(call_price_lo, S, K_call_lo, T_lo, otype="call")

            put_iv_hi  = rf.calculate(put_price_hi,  S, K_put_hi,  T_hi, otype="put")
            call_iv_hi = rf.calculate(call_price_hi, S, K_call_hi, T_hi, otype="call")

            put_iv = linear_interpolated_iv_v1(put_iv_lo, put_iv_hi, T_lo, T_hi, target_dte/365)
            call_iv = linear_interpolated_iv_v1(call_iv_lo, call_iv_hi, T_lo, T_hi, target_dte/365)

            K_put_star  = (1 - moneyness) * S
            K_call_star = (1 + moneyness) * S
            denom = (K_put_star - K_call_star) / S   # = -2*moneyness
            skew = (put_iv - call_iv) / denom
            implied_skews.append(skew)
            dates.append(chain.close_date)

    return implied_skews, dates