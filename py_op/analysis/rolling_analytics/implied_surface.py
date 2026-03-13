import numpy as np

from py_op.data.builders.option_chain_builder import create_chain_series
from py_op.calc_engine.vol_engine.iv_calc import RootFinder
"""
(3/1/2026) We will move the skew and constant maturity IV's from implied_values.py into here
"""

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
    def __init__(self, ticker: str, start_date: str, end_date:str, steps: int = 1) -> None:
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.steps = steps
        self.chain_series = create_chain_series(ticker, start_date, end_date, steps=steps)

    def implied_skew_moneyness(self, target_dte: int, moneyness: float=0.1, max_days_diff: int=20):
        """
        This is the version of implied skew from Euan Sinclair retail option trading. It is also number 6 in the flash cards ipynb
        The issue with this is we need to make it constant maturity right now it just fetches the maturity closest to dte
        Note: (2/16/26) we will change this so it gets a constant maturity dte not the closest
        """
        implied_skews, dates = [], []
        rf = RootFinder()

        for chain in self.chain_series:
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
    
    def implied_skew_fixed_strike(self, target_dte: int, moneyness: float = 0.1, max_days_diff: int = 20):
        """
        This measures 90% put iv - 110% call iv -- we should make a modification that allows 100% atm iv.
        Collin Bennet refers to 90% put iv - 100% atm iv as fixed strike skew.
        """
        implied_skews, dates = [], []
        rf = RootFinder()

        for chain in self.chain_series:
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

class RollingKurtosis:
    pass

class RollingIV:
    pass