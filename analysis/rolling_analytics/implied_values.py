import numpy as np

from calc_engine.vol_engine.iv_calc import RootFinder
from calc_engine.vol_engine.interpolation_models import linear_interpolated_iv_v1
from calc_engine.misc_funcs.put_call_parity import implied_rate
from data.builders.option_chain_builder import create_chain_series
from utils.date_utils import find_bracketing_dtes

"""
This is used to calculate all implied values 
EX:
 -- Interpolated constant maturity IV
 -- Implied skew 
 -- implied parameters from models like SABR, Heston, rBergomi
"""

def implied_skew_moneyness(ticker, start_date, end_date, target_dte, moneyness=0.1, steps=1, max_days_diff=20):
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