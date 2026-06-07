import numpy as np

from py_op.data.builders.option_chain_builder import create_chain_series
from py_op.calc_engine.vol_engine.iv_calc import InverseGaussian
from py_op.utils.date_utils import find_bracketing_dtes
from py_op.calc_engine.misc_funcs.put_call_parity import implied_rate
from py_op.calc_engine.vol_engine.vol_funcs import variance_swap_fixed_leg, linear_interpolated_iv_v1, forward_volatility
from py_op.calc_engine.greeks.analytical_greeks import AnalyticalDelta
from py_op.calc_engine.vol_engine.iv_calc import SkewCalculator

"""
(3/1/2026) We will move the skew and constant maturity IV's from implied_values.py into here
"""

class RollingAnalytics:

    def __init__(self, ticker: str, start_date: str, end_date:str, steps: int = 1, iv_calc = InverseGaussian()) -> None:
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
    def __init__(self, ticker: str, start_date: str, end_date: str, steps: int = 1, iv_calc = InverseGaussian()) -> None:
        super().__init__(ticker, start_date, end_date, steps, iv_calc)

    def _interpolated_variance_swap_fixed_leg(self, chain, target_dte: int, r: float = 0):
        S = chain.S
        dte_list = chain.get_common_dtes()
        dte_lo, dte_hi = find_bracketing_dtes(dte_list, target_dte)

        if dte_lo is None and dte_hi is None:
            return None

        if dte_hi is None:
            put_prices, call_prices, strikes, actual_dtes = chain.get_equal_skew_prices(dte=dte_lo, max_days_diff=0)
            return variance_swap_fixed_leg(S, put_prices, call_prices, strikes, actual_dtes[0], r)

        put_prices_lo, call_prices_lo, strikes_lo, actual_dtes_lo = chain.get_equal_skew_prices(dte=dte_lo, max_days_diff=0)
        put_prices_hi, call_prices_hi, strikes_hi, actual_dtes_hi = chain.get_equal_skew_prices(dte=dte_hi, max_days_diff=0)

        var_lo = variance_swap_fixed_leg(S, put_prices_lo, call_prices_lo, strikes_lo, actual_dtes_lo[0], r)
        var_hi = variance_swap_fixed_leg(S, put_prices_hi, call_prices_hi, strikes_hi, actual_dtes_hi[0], r)

        T_lo = actual_dtes_lo[0] / 365
        T_hi = actual_dtes_hi[0] / 365
        T_target = target_dte / 365

        return linear_interpolated_iv_v1(var_lo, var_hi, T_lo, T_hi, T_target)

    def _variance_swap_fixed_leg_generator(self, dte1: int = 30, dte2: int = 60, r: float = 0):
        """
        Yields one observation at a time:
            (close_date, var_swap1, var_swap2)
        where var_swap1 and var_swap2 are constant-maturity variance swaps.
        """
        for chain in self.chain_series:
            var_swap1 = self._interpolated_variance_swap_fixed_leg(chain, dte1, r)
            var_swap2 = self._interpolated_variance_swap_fixed_leg(chain, dte2, r)

            if var_swap1 is None or var_swap2 is None:
                continue

            yield chain.close_date, var_swap1, var_swap2

    def variance_swap_relative_term_premia(self, dte1: int = 30, dte2: int = 60, r: float = 0, normalized: bool = False):
        relative_premias, dates = [], []
        T1, T2 = dte1/364, dte2/364

        for date, var_swap1, var_swap2 in self._variance_swap_fixed_leg_generator(dte1, dte2, r):

            if normalized:

                term_premia = ((var_swap2 - var_swap1) / var_swap1) * (np.sqrt(T2*T1) / (np.sqrt(T2) - np.sqrt(T1)))
            else:

                term_premia = (var_swap2 - var_swap1) / var_swap1

            relative_premias.append(term_premia)
            dates.append(date)

        return relative_premias, dates
    
    def variance_swap_term_premia(self, dte1: int = 30, dte2: int = 60, r: float = 0, normalized: bool = False):
        """
        We divide by 365, but we might actually have to divide by 252.
        Normalizing the term structure makes it easier to compare if we plug in multiple dtes
        """
        term_premias, dates = [], []
        T1, T2 = dte1/364, dte2/364

        for date, var_swap1, var_swap2 in self._variance_swap_fixed_leg_generator(dte1, dte2, r):

            if normalized:

                term_premia = (var_swap2 - var_swap1) * (np.sqrt(T2*T1) / (np.sqrt(T2) - np.sqrt(T1)))
            else:

                term_premia = var_swap2 - var_swap1

            term_premias.append(term_premia)
            dates.append(date)

        return term_premias, dates

    def variance_swap_forward(self, dte1: int = 30, dte2: int = 60, r: float = 0):
        """
        
        """
        forward_factor, dates = [], []

        for date, var_swap1, var_swap2 in self._variance_swap_fixed_leg_generator(dte1, dte2, r):
            forward_vol = forward_volatility(var_swap1, var_swap2, dte1, dte2)
            forward_factor.append(forward_vol)
            dates.append(date)

        return forward_factor, dates
    
    def variance_swap_forward_factor(self, dte1: int = 30, dte2: int = 60, r: float = 0):
        """
        
        """
        forward_factor, dates = [], []

        for date, var_swap1, var_swap2 in self._variance_swap_fixed_leg_generator(dte1, dte2, r):
            forward_vol = forward_volatility(var_swap1, var_swap2, dte1, dte2)
            forward_factor.append(var_swap1/forward_vol)
            dates.append(date)

        return forward_factor, dates

    def variance_swap_z_score(self, dte1: int = 30, dte2: int = 90, window: int = 10, r: float = 0):
        term_premias, dates = self.variance_swap_term_premia(dte1, dte2, r, False)
        term_premias = np.array(term_premias)
        z_scores = []

        for i in range(len(term_premias) - window):
            rolling_mean = np.mean(term_premias[i:i + window])
            rolling_std_dev = np.std(term_premias[i:i + window])
            z_score = (term_premias[i + window] - rolling_mean) / rolling_std_dev
            z_scores.append(z_score)

        return z_scores, dates[window:]

    def atmf():
        pass

class RollingSkew(RollingAnalytics):
    """
    We need to work on the methods involved with delta in this class, we should try using skew_calculator.equal_parity_ivs, instead of otm ivs
    """

    def __init__(self, ticker: str, start_date: str, end_date: str, steps: int = 1, iv_calc=InverseGaussian(), parity_iv_relation = False, use_otm_prices = False) -> None:
        
        self.parity_iv_relation = parity_iv_relation
        self.use_otm_prices = use_otm_prices
        
        if self.use_otm_prices == True and self.parity_iv_relation == True:
            raise ValueError("Cannot use otm prices flag with equal iv skew flag")
        
        super().__init__(ticker, start_date, end_date, steps, iv_calc)
        self.delta_calc = AnalyticalDelta()
        self.skew_calculator = SkewCalculator()

    def _select_skew_points(self, target_dte: int, max_days_diff: int = 20, r: float = 0.04, q: float = 0.0, mode: str = "moneyness", put_moneyness: float = 0.1, call_moneyness: float = 0.1, put_delta: float = -0.25, call_delta: float = 0.25):
        """
        Selects the put/call IV points used to define skew from the raw market IV curve.
        """

        for chain in self.chain_series:
            
            S = chain.S
            date = chain.close_date

            if self.use_otm_prices == True:
                otm_prices, strikes, actual_dte = chain.get_otm_skew_prices(dte=target_dte, max_days_diff=max_days_diff)
                #put_prices, call_prices = np.array([price for price, K in zip(otm_prices, strikes) if K < S else 0]), np.array([price for price, K in zip(otm_prices, strikes) if K > S else 0]) 
                put_prices, call_prices = otm_prices, otm_prices

            else:

                put_prices, call_prices, strikes, actual_dte = chain.get_equal_skew_prices(dte=target_dte, max_days_diff=max_days_diff)
                actual_dte = actual_dte

            strikes = np.array(strikes, dtype=float)

            if mode == "moneyness":
                moneyness_arr = strikes / S 

                idx_put  = np.abs(moneyness_arr - (1 - put_moneyness)).argmin()
                idx_call = np.abs(moneyness_arr - (1 + call_moneyness)).argmin()

                put_price,  K_put  = float(put_prices[idx_put]),  float(strikes[idx_put])
                put_price_for_call,  K_put_for_call  = float(put_prices[idx_call]),  float(strikes[idx_call])

                call_price, K_call = float(call_prices[idx_call]), float(strikes[idx_call])
                call_price_for_put, K_call_for_put = float(call_prices[idx_put]), float(strikes[idx_put])

                if self.parity_iv_relation == True:

                    put_i_rate = implied_rate(call_price_for_put, put_price, S, K_put, actual_dte/365)
                    call_i_rate = implied_rate(call_price, put_price_for_call, S, K_call, actual_dte/365)

                    put_iv  = self.iv_calc.calculate(put_price,  S, K_put,  actual_dte/365, otype="put", r = put_i_rate)
                    call_iv = self.iv_calc.calculate(call_price, S, K_call, actual_dte/365, otype="call", r = call_i_rate)

                else:

                    put_iv  = self.iv_calc.calculate(put_price,  S, K_put,  actual_dte/365, otype="put")
                    call_iv = self.iv_calc.calculate(call_price, S, K_call, actual_dte/365, otype="call")

            elif mode == "delta":

                if self.use_otm_prices == True:
                    ivs, new_strikes = self.skew_calculator.calculate_otm_skew(S, otm_prices, strikes, actual_dte/365, r, q)
                    ivs, new_strikes = np.array(ivs), np.array(new_strikes)

                    deltas = np.where(strikes > S, self.delta_calc.calculate(S, new_strikes, actual_dte/365, ivs, r, q, otype="call"), self.delta_calc.calculate(S, new_strikes, actual_dte/365, ivs, r, q, otype="put"))

                else:
                    ivs, new_strikes = zip(*self.skew_calculator.calculate_parity_skew(S, put_prices, call_prices, strikes, actual_dte/365, r, q))
                    ivs, new_strikes = np.array(ivs), np.array(new_strikes)

                    deltas = np.where(new_strikes > S, self.delta_calc.calculate(S, new_strikes, actual_dte/365, ivs, r, q, otype="call"), self.delta_calc.calculate(S, new_strikes, actual_dte/365, ivs, r, q, otype="put"))

                idx_put  = np.abs(deltas - put_delta).argmin()
                idx_call = np.abs(deltas - call_delta).argmin()

                put_iv,  K_put  = float(ivs[idx_put]),  float(new_strikes[idx_put])
                call_iv, K_call = float(ivs[idx_call]), float(new_strikes[idx_call])

            else:
                raise ValueError("mode must be 'moneyness' or 'delta'")

            yield put_iv, call_iv, K_put, K_call, S, date

    def implied_skew_moneyness(self, target_dte: int, max_days_diff: int = 20, put_moneyness: float = 0.1, call_moneyness: float = 0.1):
        implied_skews, dates = [], []

        for put_iv, call_iv, K_put, K_call, S, date in self._select_skew_points(target_dte, max_days_diff=max_days_diff, mode="moneyness", put_moneyness=put_moneyness, call_moneyness=call_moneyness):
            
            dates.append(date)
            denom = (K_put - K_call) / S
            implied_skews.append((put_iv - call_iv) / denom)

        return implied_skews, dates

    def implied_skew_fixed_strike_moneyness(self, target_dte: int, max_days_diff: int = 20, put_moneyness: float = 0.1, call_moneyness: float = 0.1):
        implied_skews, dates = [], []

        for put_iv, call_iv, _, _, _, date in self._select_skew_points(target_dte, max_days_diff=max_days_diff, mode="moneyness", put_moneyness=put_moneyness, call_moneyness=call_moneyness):
            dates.append(date)
            implied_skews.append(put_iv - call_iv)

        return implied_skews, dates

    def implied_skew_delta(self, target_dte: int, max_days_diff: int = 20, r: float = 0.04, q: float = 0.0, put_delta: float = -0.25, call_delta: float = 0.25):
        implied_skews, dates = [], []

        for put_iv, call_iv, K_put, K_call, spot, date in self._select_skew_points(target_dte, max_days_diff=max_days_diff, r=r, q=q, mode="delta", put_delta=put_delta, call_delta=call_delta):
            dates.append(date)
            denom = (K_put - K_call) / spot
            implied_skews.append((put_iv - call_iv) / denom)

        return implied_skews, dates

    def implied_skew_fixed_strike_delta(self, target_dte: int, max_days_diff: int = 20, r: float = 0.04, q: float = 0.0, put_delta: float = -0.25, call_delta: float = 0.25):
        implied_skews, dates = [], []

        for put_iv, call_iv, _, _, _, date in self._select_skew_points(target_dte, max_days_diff=max_days_diff, r=r, q=q, mode="delta", put_delta=put_delta, call_delta=call_delta):

            dates.append(date)
            implied_skews.append(put_iv - call_iv)

        return implied_skews, dates

    def implied_skew_constant_strike(self, delta=None, moneyness=None):
        """
        Define an initial delta or moneyness at the first day of the period to set the
        strikes, then keep those strikes fixed through time.
        """
        pass

class RollingKurtosis:
    pass

class RollingVolatility(RollingAnalytics):

    def __init__(self, ticker: str, start_date: str, end_date: str, steps: int = 1, iv_calc = InverseGaussian()) -> None:
        super().__init__(ticker, start_date, end_date, steps, iv_calc)

    def constant_maturity_atm_iv(self, target_dte: int, q: float = 0.0):
        """
        This function uses linear interpolation to get a atm constant matuity iv.
        Right now it only gets put iv, put it uses an implied rate so they should be equal, but in the future we will test for call iv.
        We will make two other functions one that gets atm put iv and one that gets atm call iv.
        We should add max_days_diff = 0
        """
        ivs, dates, spots = [], [], []
        t_target = target_dte / 365

        for chain in self.chain_series:
            dte_list = chain.get_common_dtes()
            dte_lo, dte_hi = find_bracketing_dtes(dte_list, target_dte)

            S = chain.S
            spots.append(S)

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

        return ivs, dates, spots
    
    def fixed_strike_constant_exp_atm_iv(self, exp: str = None, dte: int = None, max_diff_days: int = 5, q: float = 0):
        """
        This function gets atm iv at an initial date and then gets the same strike iv going forward.
        It is important to note this returns a time series where the initial first iv is atm, but the second value/iv in the series will not be
          because spot most likely deviated from the atm strike.
        Note: For anything fixed strike it is best to use exp
        """
        if dte is not None:
            exp_data = self.chain_series[0].get_exp_from_dte(mat_days=dte, max_diff_days = max_diff_days)
            exp = exp_data[0]

        #assert exp >= self.end_date, ("Expiration date must be after end_date")

        S = self.chain_series[0].S
        ivs, dates, spots = [], [], []
        idx = 0

        #for chain in self.chain_series:
        while idx < len(self.chain_series) and self.chain_series[idx].close_date <= exp:
            cur_chain = self.chain_series[idx]

            call = cur_chain.get_option(S, otype="call", exp = exp)
            put = cur_chain.get_option(S, otype="put", exp = exp)

            real_dte = put.dte
            T = real_dte/365

            i_rate = implied_rate(call.price, put.price, S, put.strike, T)
            iv = self.iv_calc.calculate(put.price, S, put.strike, T, r=i_rate, otype="put", q=q)
            ivs.append(iv)
            dates.append(cur_chain.close_date)
            spots.append(cur_chain.S)
            idx += 1
        
        return ivs, dates, spots
    
    def fixed_strike_constant_exp_iv(self, strike: int, exp: str = None, dte: int = None, max_diff_days: int = 5, r: float = 0.00, q: float = 0):
        """
        This function gives us iv at a certain strike over time where the dte changes, ex) t1 dte = 30, t2 = 29
        So it gives us the iv for a certain option contract over time.
        """
        if dte is not None:
            exp_data = self.chain_series[0].get_exp_from_dte(mat_days=dte, max_diff_days = max_diff_days)
            exp = exp_data[0]

        ivs, dates, spots = [], [], []
        idx = 0

        #for chain in self.chain_series:
        while idx < len(self.chain_series) and self.chain_series[idx].close_date <= exp:
            cur_chain = self.chain_series[idx]

            call = cur_chain.get_option(strike, otype="call", exp = exp)
            put = cur_chain.get_option(strike, otype="put", exp = exp)

            real_dte = put.dte
            T = real_dte/365

            if put.strike == strike and call.strike == strike:
                call_price, put_price = call.price, put.price

            elif put.strike == strike:
                put_price = put.price
                call_price = put.price + cur_chain.S - strike*np.exp(-r*T)

            elif call.strike == strike:
                call_price = call.price
                put_price = call.price - cur_chain.S + strike*np.exp(-r*T)

            else:
                raise ValueError(f"Strike {strike} does not exist for exp {exp} for call or put for start date {self.start_date}")

            i_rate = implied_rate(call_price, put_price, cur_chain.S, strike, T)
            iv = self.iv_calc.calculate(put_price, cur_chain.S, strike, T, r=i_rate, otype="put", q=q)
            ivs.append(iv)
            dates.append(cur_chain.close_date)
            spots.append(cur_chain.S)
            idx += 1
        
        return ivs, dates, spots
    

    def fixed_strike_constant_maturity_iv(self, strike: int, target_dte: int, r: float = 0, q: float = 0):
        ivs, dates, spots = [], [], []
        t_target = target_dte / 365

        for chain in self.chain_series:
            dte_list = chain.get_common_dtes()
            dte_lo, dte_hi = find_bracketing_dtes(dte_list, target_dte)

            S = chain.S
            spots.append(S)

            # guard: if chain has no usable DTEs
            if dte_lo is None and dte_hi is None:
                continue

            # exact match or clamped
            if dte_hi is None:

                call = chain.get_option(strike, otype="call", dte=dte_lo, max_days_diff = 0)
                put  = chain.get_option(strike, otype="put",  dte=dte_lo, max_days_diff = 0)

                if call.strike == strike and put.strike == strike:
                    call_price = call.price
                    put_price = put.price

                elif put.strike == strike:
                    put_price = put.price
                    call_price = put.price + S - strike*np.exp(-r*T1)

                elif call.strike == strike:
                    call_price = call.price
                    put_price = call.price - S + strike*np.exp(-r*T1)

                T = put.dte / 365  # use the contract's DTE to be consistent with your objects
                i_rate = implied_rate(call_price, put_price, S, strike, T)

                iv = self.iv_calc.calculate(put_price, S, strike, T, r=i_rate, otype="put", q=q)

            # interpolate between two expiries
            else:

                call_lo = chain.get_option(strike, otype="call", dte=dte_lo, max_days_diff = 0)
                put_lo = chain.get_option(strike, otype="put", dte=dte_lo, max_days_diff = 0)
                T1 = put_lo.dte / 365

                if call_lo.strike == strike and put_lo.strike == strike:
                    call_lo_price = call_lo.price
                    put_lo_price = put_lo.price

                elif put_lo.strike == strike:
                    put_lo_price = put_lo.price
                    call_lo_price = put_lo.price + S - strike*np.exp(-r*T1)

                elif call_lo.strike == strike:
                    call_lo_price = call_lo.price
                    put_lo_price = call_lo.price - S + strike*np.exp(-r*T1)

                call_hi = chain.get_option(strike, otype="call", dte=dte_hi, max_days_diff = 0)
                put_hi = chain.get_option(strike, otype="put", dte=dte_hi, max_days_diff = 0)
                T2 = put_hi.dte / 365

                if call_hi.strike == strike and put_hi.strike == strike:
                    call_hi_price = call_hi.price
                    put_hi_price = put_hi.price

                elif put_hi.strike == strike:
                    put_hi_price = put_hi.price
                    call_hi_price = put_hi.price + S - strike*np.exp(-r*T2)

                elif call_hi.strike == strike:
                    call_hi_price = call_hi.price
                    put_hi_price = call_hi.price - S + strike*np.exp(-r*T2)

                i_rate1 = implied_rate(call_lo_price, put_lo_price, S, strike, T1)
                i_rate2 = implied_rate(call_hi_price, put_hi_price, S, strike, T2)

                iv1 = self.iv_calc.calculate(put_lo_price, S, strike, T1, r=i_rate1, otype="put", q=q)
                iv2 = self.iv_calc.calculate(put_hi_price, S, strike, T2, r=i_rate2, otype="put", q=q)

                t1 = dte_lo / 365
                t2 = dte_hi / 365
                iv = linear_interpolated_iv_v1(iv1, iv2, t1, t2, t_target)

            ivs.append(iv)
            dates.append(chain.close_date)

        return ivs, dates, spots


    def constant_maturity_variance_swap_fixed_leg(self, target_dte: int, r: float = 0):

        var_swaps, dates, spots = [], [], []

        for chain in self.chain_series:
            S = chain.S
            spots.append(S)
            dates.append(chain.close_date)
            dte_list = chain.get_common_dtes()
            dte_lo, dte_hi = find_bracketing_dtes(dte_list, target_dte)

            if dte_lo is None and dte_hi is None:
                return None

            if dte_hi is None:
                put_prices, call_prices, strikes, actual_dtes = chain.get_equal_skew_prices(dte=dte_lo, max_days_diff=0)
                interpolated_var_swap = variance_swap_fixed_leg(S, put_prices, call_prices, strikes, actual_dtes, r)

            else:

                put_prices_lo, call_prices_lo, strikes_lo, actual_dtes_lo = chain.get_equal_skew_prices(dte=dte_lo, max_days_diff=0)
                put_prices_hi, call_prices_hi, strikes_hi, actual_dtes_hi = chain.get_equal_skew_prices(dte=dte_hi, max_days_diff=0)
                var_lo = variance_swap_fixed_leg(S, put_prices_lo, call_prices_lo, strikes_lo, actual_dtes_lo, r)
                var_hi = variance_swap_fixed_leg(S, put_prices_hi, call_prices_hi, strikes_hi, actual_dtes_hi, r)

                T_lo = actual_dtes_lo / 365
                T_hi = actual_dtes_hi / 365
                T_target = target_dte / 365

                interpolated_var_swap = linear_interpolated_iv_v1(var_lo, var_hi, T_lo, T_hi, T_target)

            var_swaps.append(interpolated_var_swap)

        return var_swaps, dates, spots


"""
Old Code

"""

# We are keeping the old rolling skew because I need to update the new one so it uses constant maturity as well

class RollingSkewOld(RollingAnalytics):

    def __init__(self, ticker: str, start_date: str, end_date: str, steps: int = 1, iv_calc = InverseGaussian()) -> None:
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
    
    def implied_skew_fixed_strike(self, target_dte: int, put_moneyness: float = 0.1, call_moneyness = 0.1, max_days_diff: int = 20):
        """
        This measures 90% put iv - 110% call iv -- we should make a modification that allows 100% atm iv.
        Collin Bennet refers to 90% put iv - 100% atm iv as fixed strike skew.
        Morgan Stanley calls this relative skew, so we might change the name to relative skew.
        """
        implied_skews, dates = [], []

        for chain in self.chain_series:
            S = chain.S

            otm_prices, strikes, actual_dte = chain.get_otm_skew_prices(
                dte=target_dte, max_days_diff=max_days_diff
            )

            strikes = np.array(strikes, dtype=float)
            m_arr = strikes / S

            idx_put  = np.abs(m_arr - (1 - put_moneyness)).argmin()
            idx_call = np.abs(m_arr - (1 + call_moneyness)).argmin()

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
        rf = InverseGaussian()
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

    def implied_skew_constant_strike(self):
        """
        This method plot the rolling skew metric for two specific strikes over time eg 480 K put and 520 K call, over time.
        This is true fixed strike skew unlike the Collin Bennet version, ie the strikes do not change over time.
        """
        pass