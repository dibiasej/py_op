import numpy as np

from py_op.data.builders.option_chain_builder import create_chain_series
from py_op.calc_engine.vol_engine.iv_calc import RootFinder
from py_op.utils.date_utils import find_bracketing_dtes
from py_op.calc_engine.vol_engine.vol_funcs import linear_interpolated_iv_v1
from py_op.calc_engine.misc_funcs.put_call_parity import implied_rate
from py_op.calc_engine.vol_engine.vol_funcs import variance_swap_approximation

"""
(3/1/2026) We will move the skew and constant maturity IV's from implied_values.py into here
"""

class RollingAnalytics:

    def __init__(self, ticker: str, start_date: str, end_date:str, steps: int = 1, iv_calc = RootFinder()) -> None:
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.steps = steps
        self.iv_calc = iv_calc
        self.chain_series = create_chain_series(ticker, start_date, end_date, steps=steps)

class RollingTermStructure(RollingAnalytics):
    """
    This class will be used to get rolling term structure analytics.
    Either add argument to each method that says z_score = True, or make completely seperate methods eg, atmf_zscore.
    I think we will force all of them to use a constant maturity
    Things to add (For each var swap, atmf, constant maturity)
    - z-score (Euan Sinclair)
    - Relative term premia (Benn)
    - Term premia
    - Forward Factor (Jarod)
    We have a lot of the same types of calculations in this, ex relative term premia and regular both use the same exact logic the ending formula is the only thing than changes
    so we should try to make a simpler version
    """
    def __init__(self, ticker: str, start_date: str, end_date: str, steps: int = 1, iv_calc = RootFinder()) -> None:
        super().__init__(ticker, start_date, end_date, steps, iv_calc)

    def _interpolated_variance_swap(self, chain, target_dte: int, r: float = 0):
        S = chain.S
        dte_list = chain.get_common_dtes()
        dte_lo, dte_hi = find_bracketing_dtes(dte_list, target_dte)

        if dte_lo is None and dte_hi is None:
            return None

        if dte_hi is None:
            put_prices, call_prices, strikes, actual_dtes = chain.get_equal_skew_prices(dte=dte_lo, max_days_diff=0)
            return variance_swap_approximation(S, put_prices, call_prices, strikes, actual_dtes[0], r)

        put_prices_lo, call_prices_lo, strikes_lo, actual_dtes_lo = chain.get_equal_skew_prices(dte=dte_lo, max_days_diff=0)
        put_prices_hi, call_prices_hi, strikes_hi, actual_dtes_hi = chain.get_equal_skew_prices(dte=dte_hi, max_days_diff=0)

        var_lo = variance_swap_approximation(S, put_prices_lo, call_prices_lo, strikes_lo, actual_dtes_lo[0], r)
        var_hi = variance_swap_approximation(S, put_prices_hi, call_prices_hi, strikes_hi, actual_dtes_hi[0], r)

        T_lo = actual_dtes_lo[0] / 365
        T_hi = actual_dtes_hi[0] / 365
        T_target = target_dte / 365

        return linear_interpolated_iv_v1(var_lo, var_hi, T_lo, T_hi, T_target)

    def _variance_swap_generator(self, dte1: int = 30, dte2: int = 60, r: float = 0):
        """
        Yields one observation at a time:
            (close_date, var_swap1, var_swap2)
        where var_swap1 and var_swap2 are constant-maturity variance swaps.
        """
        for chain in self.chain_series:
            var_swap1 = self._interpolated_variance_swap(chain, dte1, r)
            var_swap2 = self._interpolated_variance_swap(chain, dte2, r)

            if var_swap1 is None or var_swap2 is None:
                continue

            yield chain.close_date, var_swap1, var_swap2

    def variance_swap_relative_term_premia(self, dte1: int = 30, dte2: int = 60, r: float = 0):
        relative_premias, dates = [], []

        for date, var_swap1, var_swap2 in self._variance_swap_generator(dte1, dte2, r):
            term_premia = (var_swap2 - var_swap1) / var_swap1
            relative_premias.append(term_premia)
            dates.append(date)

        return relative_premias, dates
    
    def variance_swap_term_premia(self, dte1: int = 30, dte2: int = 60, r: float = 0):
        term_premias, dates = [], []

        for date, var_swap1, var_swap2 in self._variance_swap_generator(dte1, dte2, r):
            term_premia = var_swap2 - var_swap1
            term_premias.append(term_premia)
            dates.append(date)

        return term_premias, dates
    
    def variance_swap()

    def atmf():
        pass


class RollingSkew(RollingAnalytics):

    def __init__(self, ticker: str, start_date: str, end_date: str, steps: int = 1, iv_calc = RootFinder()) -> None:
        super().__init__(ticker, start_date, end_date, steps, iv_calc)

    def implied_skew_moneyness(self, target_dte: int, moneyness: float=0.1, max_days_diff: int=20):
        """
        This is the version of implied skew from Euan Sinclair retail option trading. It is also number 6 in the flash cards ipynb
        The issue with this is we need to make it constant maturity right now it just fetches the maturity closest to dte
        Note: (2/16/26) we will change this so it gets a constant maturity dte not the closest
        """
        implied_skews, dates = [], []

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

            put_iv  = self.iv_calc.calculate(put_price,  S, K_put,  T, otype="put")
            call_iv = self.iv_calc.calculate(call_price, S, K_call, T, otype="call")

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

            put_iv  = self.iv_calc.calculate(put_price,  S, K_put,  T, otype="put")
            call_iv = self.iv_calc.calculate(call_price, S, K_call, T, otype="call")

            implied_skews.append(put_iv - call_iv)
            dates.append(chain.close_date)

        return implied_skews, dates

    def implied_skew_moneyness_constant_maturity(self, target_dte: int, moneyness: float = .1):
        """
        This gets the constant maturity 90% - 110% skew, the value is very similar to the implied_skew_moneyness so I might get rid of it
        """
        rf = RootFinder()
        implied_skews, dates = [], []

        for chain in self.chain_series:
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

                put_iv  = self.iv_calc.calculate(put_price,  S, K_put,  T, otype="put")
                call_iv = self.iv_calc.calculate(call_price, S, K_call, T, otype="call")

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

                put_iv_lo  = self.iv_calc.calculate(put_price_lo,  S, K_put_lo,  T_lo, otype="put")
                call_iv_lo = self.iv_calc.calculate(call_price_lo, S, K_call_lo, T_lo, otype="call")

                put_iv_hi  = self.iv_calc.calculate(put_price_hi,  S, K_put_hi,  T_hi, otype="put")
                call_iv_hi = self.iv_calc.calculate(call_price_hi, S, K_call_hi, T_hi, otype="call")

                put_iv = linear_interpolated_iv_v1(put_iv_lo, put_iv_hi, T_lo, T_hi, target_dte/365)
                call_iv = linear_interpolated_iv_v1(call_iv_lo, call_iv_hi, T_lo, T_hi, target_dte/365)

                K_put_star  = (1 - moneyness) * S
                K_call_star = (1 + moneyness) * S
                denom = (K_put_star - K_call_star) / S   # = -2*moneyness
                skew = (put_iv - call_iv) / denom
                implied_skews.append(skew)
                dates.append(chain.close_date)

        return implied_skews, dates

    def implied_skew_fixed_strike_constant_maturity(self, target_dte: int, moneyness: float = 0.1):
        """
        Constant maturity fixed strike skew.

        Default version measures:
            IV(90% put, target maturity) - IV(110% call, target maturity)

        If you later want Colin Bennett style fixed strike skew, you could modify
        this to do:
            IV(90% put) - IV(100% ATM)
        """
        implied_skews, dates = [], []

        for chain in self.chain_series:
            S = chain.S
            dte_list = chain.get_common_dtes()
            dte_lo, dte_hi = find_bracketing_dtes(dte_list, target_dte)

            if dte_lo is None and dte_hi is None:
                continue

            # Case 1: only one usable maturity
            if dte_hi is None:
                otm_prices, strikes, actual_dte = chain.get_otm_skew_prices(
                    dte=dte_lo, max_days_diff=0
                )

                strikes = np.array(strikes, dtype=float)
                moneyness_arr = strikes / S

                idx_put  = np.abs(moneyness_arr - (1 - moneyness)).argmin()
                idx_call = np.abs(moneyness_arr - (1 + moneyness)).argmin()

                put_price,  K_put  = float(otm_prices[idx_put]),  float(strikes[idx_put])
                call_price, K_call = float(otm_prices[idx_call]), float(strikes[idx_call])

                T = actual_dte / 365.0

                put_iv  = self.iv_calc.calculate(put_price,  S, K_put,  T, otype="put")
                call_iv = self.iv_calc.calculate(call_price, S, K_call, T, otype="call")

                implied_skews.append(put_iv - call_iv)
                dates.append(chain.close_date)

            # Case 2: interpolate to constant maturity
            else:
                otm_prices_lo, strikes_lo, actual_dte_lo = chain.get_otm_skew_prices(
                    dte=dte_lo, max_days_diff=0
                )
                otm_prices_hi, strikes_hi, actual_dte_hi = chain.get_otm_skew_prices(
                    dte=dte_hi, max_days_diff=0
                )

                strikes_lo = np.array(strikes_lo, dtype=float)
                strikes_hi = np.array(strikes_hi, dtype=float)

                moneyness_arr_lo = strikes_lo / S
                moneyness_arr_hi = strikes_hi / S

                idx_put_lo  = np.abs(moneyness_arr_lo - (1 - moneyness)).argmin()
                idx_call_lo = np.abs(moneyness_arr_lo - (1 + moneyness)).argmin()

                idx_put_hi  = np.abs(moneyness_arr_hi - (1 - moneyness)).argmin()
                idx_call_hi = np.abs(moneyness_arr_hi - (1 + moneyness)).argmin()

                put_price_lo,  K_put_lo  = float(otm_prices_lo[idx_put_lo]),  float(strikes_lo[idx_put_lo])
                call_price_lo, K_call_lo = float(otm_prices_lo[idx_call_lo]), float(strikes_lo[idx_call_lo])

                put_price_hi,  K_put_hi  = float(otm_prices_hi[idx_put_hi]),  float(strikes_hi[idx_put_hi])
                call_price_hi, K_call_hi = float(otm_prices_hi[idx_call_hi]), float(strikes_hi[idx_call_hi])

                T_lo = actual_dte_lo / 365.0
                T_hi = actual_dte_hi / 365.0
                T_target = target_dte / 365.0

                put_iv_lo  = self.iv_calc.calculate(put_price_lo,  S, K_put_lo,  T_lo, otype="put")
                call_iv_lo = self.iv_calc.calculate(call_price_lo, S, K_call_lo, T_lo, otype="call")

                put_iv_hi  = self.iv_calc.calculate(put_price_hi,  S, K_put_hi,  T_hi, otype="put")
                call_iv_hi = self.iv_calc.calculate(call_price_hi, S, K_call_hi, T_hi, otype="call")

                put_iv = linear_interpolated_iv_v1(put_iv_lo, put_iv_hi, T_lo, T_hi, T_target)
                call_iv = linear_interpolated_iv_v1(call_iv_lo, call_iv_hi, T_lo, T_hi, T_target)

                implied_skews.append(put_iv - call_iv)
                dates.append(chain.close_date)

        return implied_skews, dates

    def implied_skew_delta(self):
        pass

class RollingKurtosis:
    pass

class RollingIV:

    def __init__(self, ticker: str, start_date: str, end_date: str, steps: int = 1, iv_calc = RootFinder()) -> None:
        super().__init__(ticker, start_date, end_date, steps, iv_calc)

    def constant_maturity_atm_iv(self, target_dte: int, q: float = 0.0):
        """
        This function uses linear interpolation to get a atm constant matuity iv.
        Right now it only gets put iv, put it uses an implied rate so they should be equal, but in the future we will test for call iv.
        We will make two other functions one that gets atm put iv and one that gets atm call iv.
        We should add max_days_diff = 0
        """
        ivs, dates = [], []
        t_target = target_dte / 365

        for chain in self.chain_series:
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

                iv = self.iv_calc.calculate(put.price, S, put.strike, T, r=i_rate, otype="put", q=q)

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

                iv1 = self.iv_calc.calculate(put_lo.price, S, put_lo.strike, T1, r=i_rate1, otype="put", q=q)
                iv2 = self.iv_calc.calculate(put_hi.price, S, put_hi.strike, T2, r=i_rate2, otype="put", q=q)

                t1 = dte_lo / 365
                t2 = dte_hi / 365
                iv = linear_interpolated_iv_v1(iv1, iv2, t1, t2, t_target)

            ivs.append(iv)
            dates.append(chain.close_date)

        return ivs, dates