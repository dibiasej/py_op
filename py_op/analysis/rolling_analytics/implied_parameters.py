import numpy as np

from py_op.data.builders.option_chain_builder import create_chain_series
from py_op.calc_engine.vol_engine.iv_calc import RootFinder
from py_op.calc_engine.vol_engine.interpolation_models import GVV
from py_op.calc_engine.vol_engine.iv_calc import SkewCalculator
from py_op.calc_engine.greeks.analytical_greeks import AnalyticalDelta

class RollingAnalytics:

    def __init__(self, ticker: str, start_date: str, end_date:str, moneyness: float = None, steps: int = 1, iv_calc = RootFinder()) -> None:
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.moneyness = moneyness
        self.steps = steps
        self.iv_calc = iv_calc
        self.chain_series = create_chain_series(ticker, start_date, end_date, moneyness=moneyness, steps=steps)

class RollingGVV(RollingAnalytics):
    # We might add term structure metrics as well eg term premia

    def __init__(self, ticker: str, start_date: str, end_date: str, moneyness: float = None, steps: int = 1, iv_calc = RootFinder()) -> None:
        super().__init__(ticker, start_date, end_date, moneyness, steps, iv_calc)
        self.gvv = GVV()
        self.skew_calculator = SkewCalculator()

    def _implied_parameter_helper(self, dte: float, r: float = 0.04, weights: bool = False):

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
            put_prices, call_prices, strikes, actual_dtes = chain.get_equal_skew_prices(dte = dte, max_days_diff=20)
            actual_dte = actual_dtes[0]
            F = S*np.exp(r * (actual_dte/365))
            parity_ivs, new_strikes = zip(*self.skew_calculator.calculate_parity_skew_all_data(F, put_prices, call_prices, strikes, actual_dte/365))
            vol_level, spot_vol_corr, vol_vol = self.gvv.implied_parameters(F, new_strikes, parity_ivs, actual_dte/365, weights)
            ivs, strikes = self.gvv.skew(F, new_strikes, parity_ivs, actual_dte/365, weights)

            close_dates.append(close_date)
            spots.append(S)

            vol_levels.append(vol_levels)
            spot_vol_corrs.append(spot_vol_corrs)
            vol_vols.append(vol_vols)

            strike_list.append(strikes)
            skews.append(ivs)

        return close_dates, spots, vol_levels, spot_vol_corrs, vol_vols, skews, strike_list

    def vol_level(self, dte: float, r: float = 0.04, weights: bool = False):
        dates, _, vol_levels, _, _, _, _ = self._implied_parameter_helper(dte, r, weights)
        return dates, vol_levels

    def spot_vol_corr(self, dte: float, r: float = 0.04, weights: bool = False):
        dates, _, _, spot_vol_corrs, _, _, _ = self._implied_parameter_helper(dte, r, weights)
        return dates, spot_vol_corrs

    def vol_vol(self, dte: float, r: float = 0.04, weights: bool = False):
        dates, _, _, _, vol_vols, _, _ = self._implied_parameter_helper(dte, r, weights)
        return dates, vol_vols

    def skew_curve(self, dte: float, r: float = 0.04, weights: bool = False):
        dates, _, _, _, _, skews, strikes = self._implied_parameter_helper(dte, r, weights)
        return dates, skews, strikes

    def implied_skew_moneyness(self, dte: float, r: float = 0.04, weights: bool = False, put_moneyness: float = .1, call_moneyness: float = .1):
        dates, spots, _, _, _, skews, strikes_list = self._implied_parameter_helper(dte, r, weights)
        implied_skews = []

        for i in range(len(skews)):
            spot = spots[i]
            skew = skews[i]
            strikes = strikes_list[i]
            moneyness_list = np.array(strikes) / spot

            idx_put  = np.abs(moneyness_list - (1 - put_moneyness)).argmin()
            idx_call = np.abs(moneyness_list - (1 + call_moneyness)).argmin()

            put_iv,  K_put  = float(skew[idx_put]),  float(strikes[idx_put])
            call_iv, K_call = float(skew[idx_call]), float(strikes[idx_call])
    
            denom = (K_put - K_call) / spot
            implied_skews.append((put_iv - call_iv) / denom)

        return implied_skews, dates

    def implied_skew_delta(self, dte: float, r: float = 0.04, weights: bool = False, put_delta: float = .1, call_delta: float = .1):

        dates, spots, _, _, _, skews, strikes_list = self._implied_parameter_helper(dte, r, weights)
        implied_skews = []

        for i in range(len(skews)):
            spot = spots[i]
            skew = np.array(skews[i])
            strikes = np.array(strikes_list[i])

            deltas = np.where(strikes > spot, AnalyticalDelta().calculate(spot, strikes, dte/365, skew, r, q, otype = "call"), AnalyticalDelta().calculate(spot, strikes, dte/365, skew, r, q, otype = "put"))

            idx_put  = np.abs(deltas - (1 - put_delta)).argmin()
            idx_call = np.abs(deltas - (1 + call_delta)).argmin()

            put_iv,  K_put  = float(skew[idx_put]),  float(strikes[idx_put])
            call_iv, K_call = float(skew[idx_call]), float(strikes[idx_call])

            denom = (K_put - K_call) / spot
            implied_skews.append((put_iv - call_iv) / denom)

        return implied_skews, dates
    
    def implied_skew_fixed_strike_moneyness(self, dte: float, r: float = 0.04, weights: bool = False, put_moneyness: float = .1, call_moneyness: float = .1):
        """
        We call it Fixed strike because this is the convention Collin Bennet uses in his book, we may change the name in the future
        Use put_moneyness .1 and call 0 to get Collin Bennets exact definition
        """
        dates, spots, _, _, _, skews, strikes_list = self._implied_parameter_helper(dte, r, weights)
        implied_skews = []

        for i in range(len(skews)):
            spot = spots[i]
            skew = np.array(skews[i])
            strikes = np.array(strikes_list[i])
            moneyness_list = np.array(strikes) / spot

            idx_put  = np.abs(moneyness_list - (1 - put_moneyness)).argmin()
            idx_call = np.abs(moneyness_list - (1 + call_moneyness)).argmin()

            deltas = np.where(strikes > spot, AnalyticalDelta().calculate(spot, strikes, dte/365, skew, r, q, otype = "call"), AnalyticalDelta().calculate(spot, strikes, dte/365, skew, r, q, otype = "put"))

            idx_put  = np.abs(deltas - (1 - put_moneyness)).argmin()
            idx_call = np.abs(deltas - (1 + call_moneyness)).argmin()

            put_iv = float(skew[idx_put])
            call_iv = float(skew[idx_call])

            implied_skews.append(put_iv - call_iv)

        return implied_skews, dates
    
    def implied_skew_fixed_strike_delta(self, dte: float, r: float = 0.04, weights: bool = False, put_delta: float = .1, call_delta: float = .1):
        dates, spots, _, _, _, skews, strikes_list = self._implied_parameter_helper(dte, r, weights)
        implied_skews = []

        for i in range(len(skews)):
            spot = spots[i]
            skew = np.array(skews[i])
            strikes = np.array(strikes_list[i])

            deltas = np.where(strikes > spot, AnalyticalDelta().calculate(spot, strikes, dte/365, skew, r, q, otype = "call"), AnalyticalDelta().calculate(spot, strikes, dte/365, skew, r, q, otype = "put"))

            idx_put  = np.abs(deltas - (1 - put_delta)).argmin()
            idx_call = np.abs(deltas - (1 + call_delta)).argmin()

            put_iv  = float(skew[idx_put])
            call_iv = float(skew[idx_call])

            implied_skews.append(put_iv - call_iv)

        return implied_skews, dates

    def implied_skew_constant_strike(self):
        pass
        

class RollingHeston:
    pass

class RollingSABR:
    pass

class RollingrBergomi:
    pass

class RollingSVI:
    pass