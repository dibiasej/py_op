import numpy as np

from py_op.data.builders.option_chain_builder import create_chain_series
from py_op.calc_engine.vol_engine.iv_calc import RootFinder, InverseGaussian
from py_op.calc_engine.vol_engine.interpolation_models import GVV
from py_op.calc_engine.vol_engine.iv_calc import SkewCalculator
from py_op.calc_engine.greeks.analytical_greeks import AnalyticalDelta

class RollingAnalytics:

    def __init__(self, ticker: str, start_date: str, end_date:str, moneyness: float = None, steps: int = 1, iv_calc = InverseGaussian()) -> None:
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.moneyness = moneyness
        self.steps = steps
        self.iv_calc = iv_calc
        self.chain_series = create_chain_series(ticker, start_date, end_date, moneyness=moneyness, steps=steps)

class RollingGVV(RollingAnalytics):
    # We might add term structure metrics as well eg term premia

    def __init__(self, ticker: str, start_date: str, end_date: str, moneyness: float = None, steps: int = 1, iv_calc = InverseGaussian()) -> None:
        super().__init__(ticker, start_date, end_date, moneyness, steps, iv_calc)
        self.gvv = GVV()
        self.skew_calculator = SkewCalculator()
        self.delta_calc = AnalyticalDelta()

    def _implied_parameter_helper(self, dte: float, r: float = 0.04, weights: bool = True):

        close_dates = []

        vol_levels = []
        spot_vol_corrs = []
        vol_vols = []

        skews = []
        strike_list = []
        spots = []

        for chain in self.chain_series:
            close_date = chain.close_date
            S = chain.S
            put_prices, call_prices, strikes, actual_dte = chain.get_equal_skew_prices(dte = dte, max_days_diff=20)
            #actual_dte = actual_dtes[0]
            #F = S*np.exp(r * (actual_dte/365))
            #parity_ivs, new_strikes = zip(*self.skew_calculator.calculate_parity_skew_all_data(F, put_prices, call_prices, strikes, actual_dte/365))
            parity_ivs, new_strikes = zip(*self.skew_calculator.calculate_parity_skew(S, put_prices, call_prices, strikes, actual_dte/365))
            vol_level, spot_vol_corr, vol_vol = self.gvv.implied_parameters(S, new_strikes, parity_ivs, actual_dte/365, weights)
            ivs, strikes = self.gvv.skew(S, new_strikes, parity_ivs, actual_dte/365, weights, method = "polynomial")

            close_dates.append(close_date)
            spots.append(S)

            vol_levels.append(vol_level)
            spot_vol_corrs.append(spot_vol_corr)
            vol_vols.append(vol_vol)

            strike_list.append(strikes)
            skews.append(ivs)

        return close_dates, spots, vol_levels, spot_vol_corrs, vol_vols, skews, strike_list

    def _select_skew_points(self, dte: float, r: float = 0.04, q: float = 0.0, weights: bool = True, mode: str = "moneyness", put_moneyness: float = 0.1, call_moneyness: float = 0.1, put_delta: float = -0.25, call_delta: float = 0.25):
        dates, spots, vol_levels, _, _, skews, strikes_list = self._implied_parameter_helper(dte, r, weights)

        for i in range(len(skews)):
            spot = spots[i]
            atm_iv = vol_levels[i]
            skew = skews[i]
            date = dates[i]
            strikes = strikes_list[i]

            if mode == "moneyness":
                moneyness_list = np.array(strikes) / spot

                idx_put  = np.abs(moneyness_list - (1 - put_moneyness)).argmin()
                idx_call = np.abs(moneyness_list - (1 + call_moneyness)).argmin()

            elif mode == "delta":
                deltas = np.where(strikes > spot, self.delta_calc.calculate(spot, strikes, dte/365, skew, r, q, otype = "call"), self.delta_calc.calculate(spot, strikes, dte/365, skew, r, q, otype = "put"))

                idx_put  = np.abs(deltas - put_delta).argmin()
                idx_call = np.abs(deltas - call_delta).argmin()

            put_iv,  K_put  = float(skew[idx_put]),  float(strikes[idx_put])
            call_iv, K_call = float(skew[idx_call]), float(strikes[idx_call])
            yield put_iv, call_iv, K_put, K_call, spot, date, atm_iv


    def vol_level(self, dte: float, r: float = 0.04, weights: bool = True):
        dates, _, vol_levels, _, _, _, _ = self._implied_parameter_helper(dte, r, weights)
        return dates, vol_levels

    def spot_vol_corr(self, dte: float, r: float = 0.04, weights: bool = True):
        dates, _, _, spot_vol_corrs, _, _, _ = self._implied_parameter_helper(dte, r, weights)
        return dates, spot_vol_corrs

    def spot_vol_cov(self, dte: float, r: float = 0.04, weights: bool = True):
        dates, _, vol_levels, spot_vol_corrs, vol_vols, _, _ = self._implied_parameter_helper(dte, r, weights)
        return dates, np.array(spot_vol_corrs) * np.array(vol_levels) * np.array(vol_vols)

    def vol_vol(self, dte: float, r: float = 0.04, weights: bool = True):
        dates, _, _, _, vol_vols, _, _ = self._implied_parameter_helper(dte, r, weights)
        return dates, vol_vols

    def skew_curve(self, dte: float, r: float = 0.04, weights: bool = True):
        dates, _, _, _, _, skews, strikes = self._implied_parameter_helper(dte, r, weights)
        return dates, skews, strikes
    
    def implied_skew_moneyness(self, dte: float, r: float = 0.04, q: float = 0, weights: bool = False, put_moneyness: float = 0.1, call_moneyness: float = 0.1):
        """
        This method measures the skew curve using otm put and call ivs at a certain moneyness, then normalizes that by dividing by the strikes.
        This is a way to approximate the ATM skew slope using a finite difference type of model.
        This is from some content somewhere that Euan Sinclair put out.
        """
        implied_skews, dates = [], []
        for put_iv, call_iv, K_put, K_call, spot, date, atm_iv in self._select_skew_points(dte, r, q, weights, mode="moneyness", put_moneyness=put_moneyness, call_moneyness=call_moneyness):
            dates.append(date)
            denom = (K_put - K_call) / spot
            implied_skews.append((put_iv - call_iv) / denom)

        return implied_skews, dates

    def implied_skew_delta(self, dte: float, r: float = 0.04, q: float = 0, weights: bool = True, put_delta: float = -0.25, call_delta: float = 0.25):

        implied_skews, dates = [], []
        for put_iv, call_iv, K_put, K_call, spot, date, atm_iv in self._select_skew_points(dte, r, q, weights, mode="delta", put_delta=put_delta, call_delta=call_delta):
            dates.append(date)
            denom = (K_put - K_call) / spot
            implied_skews.append((put_iv - call_iv) / denom)

        return implied_skews, dates

    def implied_skew_fixed_strike_moneyness(self, dte: float, r: float = 0.04, q: float = 0, weights: bool = True, put_moneyness: float = 0.1, call_moneyness: float = 0.1):
        """
        We call it Fixed strike because this is the convention Collin Bennet uses in his book, we may change the name in the future
        Use put_moneyness .1 and call 0 to get Collin Bennets exact definition
        """
        implied_skews, dates = [], []
        for put_iv, call_iv, K_put, K_call, spot, date, atm_iv in self._select_skew_points(dte, r, q, weights, mode="moneyness", put_moneyness=put_moneyness, call_moneyness=call_moneyness):
            dates.append(date)
            implied_skews.append(put_iv - call_iv)

        return implied_skews, dates

    def implied_skew_fixed_strike_delta(self, dte: float, r: float = 0.04, q: float = 0, weights: bool = True, put_delta: float = -0.25, call_delta: float = 0.25):
        """
        We call it Fixed strike because this is the convention Collin Bennet uses in his book, we may change the name in the future
        Use put_moneyness .1 and call 0 to get Collin Bennets exact definition.
        This is also how you can measure a Risk Reversal IV.
        """
        implied_skews, dates = [], []
        for put_iv, call_iv, K_put, K_call, spot, date, atm_iv in self._select_skew_points(dte, r, q, weights, mode="delta", put_delta=put_delta, call_delta=call_delta):
            dates.append(date)
            implied_skews.append(put_iv - call_iv)

        return implied_skews, dates

    def implied_skewness(self, dte: float, r: float = 0.04, q: float = 0, weights: bool = True):
        """
        This is the implied skewness/measurement of the implied volatility curve given in Euan Sinclairs retail option trading book.
        """
        implied_skews, dates, atm_ivs = [], [], []
        for put_iv, call_iv, K_put, K_call, spot, date, atm_iv in self._select_skew_points(dte, r, q, weights, mode="delta", put_delta=-0.25, call_delta=0.25):
            dates.append(date)
            implied_skews.append(4.448 * ((put_iv - call_iv) / atm_iv))
            atm_ivs.append(atm_iv)

        return implied_skews, dates

    def risk_reversal_volatility(self, dte: float, r: float = 0.04, q: float = 0, weights: bool = True, put_delta: float = -0.25, call_delta: float = 0.25):
        """
        This method uses 25 delta ivs for puts and calls but it doesnt have to it is just a good rule of thumb to use the 25 delta vols.
        This is also related to implied kurtosis
        """
        rr_vols, spots, dates = [], [], []
        for put_iv, call_iv, K_put, K_call, spot, date, atm_iv in self._select_skew_points(dte, r, q, weights, mode="delta", put_delta=put_delta, call_delta=call_delta):
            dates.append(date)
            rr_vols.append(call_iv - put_iv)
            spots.append(spot)

        return rr_vols, spots, dates

    def butterfly_volatility(self, dte: float, r: float = 0.04, q: float = 0, weights: bool = True, put_delta: float = -0.25, call_delta: float = 0.25):
        """
        This method uses 25 delta ivs for puts and calls but it doesnt have to it is just a good rule of thumb to use the 25 delta vols.
        This is also related to implied kurtosis
        The higher this is the more expensive the wings are/the greater the curvature of the curve
        """
        butterfly_vols, dates = [], []
        for put_iv, call_iv, K_put, K_call, spot, date, atm_iv in self._select_skew_points(dte, r, q, weights, mode="delta", put_delta=put_delta, call_delta=call_delta):
            dates.append(date)
            butterfly_vols.append(((call_iv + put_iv) / 2) - atm_iv)

        return butterfly_vols, dates

    def implied_kurtosis(self, dte: float, r: float = 0.04, q: float = 0, weights: bool = True, put_delta: float = -0.05, call_delta: float = 0.05):
        """
        This method is of measuring implied kurtosis directly measures how much otm puts are overpriced compared to calls using 5 delta put iv - 5 delta call iv.
        """
        implied_kurtosis, dates = [], []
        for put_iv, call_iv, K_put, K_call, spot, date, atm_iv in self._select_skew_points(dte, r, q, weights, mode="delta", put_delta=put_delta, call_delta=call_delta):
            dates.append(date)
            implied_kurtosis.append(put_iv - call_iv)

        return implied_kurtosis, dates
    
    def implied_kurtosis_normalized(self, dte: float, r: float = 0.04, q: float = 0, weights: bool = True, put_delta: float = -0.05, call_delta: float = 0.05):
        """
        This method 
        """
        implied_kurtosis, dates = [], []
        for put_iv, call_iv, K_put, K_call, spot, date, atm_iv in self._select_skew_points(dte, r, q, weights, mode="delta", put_delta=put_delta, call_delta=call_delta):
            dates.append(date)
            implied_kurtosis.append((put_iv - call_iv) / atm_iv)

        return implied_kurtosis, dates

    def implied_kurtosis_natenberg(self, dte: float, r: float = 0.04, q: float = 0, weights: bool = True, put_delta: float = -0.05, call_delta: float = 0.05):
        """
        This method is from Natenbergs book and he uses 5 delta call iv - 5 delta put iv to measure kurtosis.
        """
        implied_kurtosis, dates = [], []
        for put_iv, call_iv, K_put, K_call, spot, date, atm_iv in self._select_skew_points(dte, r, q, weights, mode="delta", put_delta=put_delta, call_delta=call_delta):
            dates.append(date)
            implied_kurtosis.append(call_iv - put_iv)

        return implied_kurtosis, dates

    def implied_skew_constant_strike(self, delta, moneyness):
        """"
        Define an initial delta or moneyness at the first day of the period to set the strikes but these strikes do not change over time.
        """
        pass

class RollingHeston:
    pass

class RollingSABR:
    pass

class RollingrBergomi:
    pass

class RollingSVI:
    pass