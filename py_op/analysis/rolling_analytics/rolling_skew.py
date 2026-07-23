import numpy as np

from py_op.data.builders.option_chain_builder import create_chain_series
from py_op.calc_engine.vol_engine.iv_calc import InverseGaussian
from py_op.calc_engine.vol_engine.skew_models import VolatilityModel
from py_op.calc_engine.vol_engine.iv_calc import SkewCalculator
from py_op.calc_engine.greeks.analytical_greeks import AnalyticalDelta

class RollingAnalytics:

    def __init__(self, ticker: str, start_date: str, end_date:str, moneyness: float = None, steps: int = 1) -> None:
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.moneyness = moneyness
        self.steps = steps
        self.chain_series = create_chain_series(ticker, start_date, end_date, moneyness=moneyness, steps=steps)


class ImpliedSkew(RollingAnalytics):
    
    def __init__(self, ticker, start_date, end_date, moneyness = None, steps = 1, model: VolatilityModel = None):
        super().__init__(ticker, start_date, end_date, moneyness, steps)
        self.model = model
        self.skew_calculator = SkewCalculator()
        self.delta_calc = AnalyticalDelta()

    def _implied_parameters(self, dte, **kwargs):
        """
        When calling this we need to remember the order of the parameters returned by each model, calling SVI and GVV will return different parameters so remember that when calling it.
        """

        for chain in self.chain_series:
            close_date = chain.close_date
            S = chain.S
            put_prices, call_prices, strikes, actual_dte = chain.get_equal_skew_prices(dte = dte, max_days_diff=20)
            #actual_dte = actual_dtes[0]
            #F = S*np.exp(r * (actual_dte/365))
            parity_ivs, new_strikes = zip(*self.skew_calculator.calculate_parity_skew(S, put_prices, call_prices, strikes, actual_dte/365))
            params = self.model.implied_parameters(S, new_strikes, parity_ivs, actual_dte/365, **kwargs)

            yield close_date, S, params

    def _fetch_skew_points(self, S, skew, strikes, dte: float, r: float = 0.04, q: float = 0.0, mode: str = "moneyness", put_moneyness: float = 0.1, call_moneyness: float = 0.1, put_delta: float = -0.25, call_delta: float = 0.25):

        if mode == "moneyness":
            moneyness_list = np.array(strikes) / S

            idx_put  = np.abs(moneyness_list - (1 - put_moneyness)).argmin()
            idx_call = np.abs(moneyness_list - (1 + call_moneyness)).argmin()
            idx_atm = np.abs(moneyness_list - 1).argmin()

        elif mode == "delta":
            deltas = np.where(strikes > S, self.delta_calc.calculate(S, strikes, dte/365, skew, r, q, otype = "call"), self.delta_calc.calculate(S, strikes, dte/365, skew, r, q, otype = "put"))
            print(f"deltas: {deltas}\n")
            idx_put  = np.abs(deltas - put_delta).argmin()
            idx_call = np.abs(deltas - call_delta).argmin()
            idx_atm = np.abs(deltas - (-.5)).argmin() # this is the idx for an atm put

        put_iv,  K_put  = float(skew[idx_put]),  float(strikes[idx_put])
        call_iv, K_call = float(skew[idx_call]), float(strikes[idx_call])
        atm_iv, K_atm = float(skew[idx_atm]), float(strikes[idx_atm])
        return put_iv, call_iv, atm_iv, K_put, K_call, K_atm

    def _skew_data_generator(self, dte: float, r: float = 0.04, q: float = 0.0, mode: str = "moneyness", put_moneyness: float = 0.1, call_moneyness: float = 0.1, put_delta: float = -0.25, call_delta: float = 0.25, **kwargs):
        """
        dte passed into this method should not be divided by 365
        """
        for chain in self.chain_series:
            close_date = chain.close_date
            S = chain.S
            put_prices, call_prices, strikes, actual_dte = chain.get_equal_skew_prices(dte = dte, max_days_diff=20)
            parity_ivs, new_strikes = zip(*self.skew_calculator.calculate_parity_skew(S, put_prices, call_prices, strikes, actual_dte/365))

            if self.model is None or self.model == "market":
                params = None
                ivs = parity_ivs
                selected_strikes = new_strikes
            
            else:
                params = self.model.implied_parameters(S, new_strikes, parity_ivs, actual_dte/365, **kwargs)
                ivs, selected_strikes = self.model.skew(S, new_strikes, parity_ivs, actual_dte/365, **kwargs)

            put_iv, call_iv, atm_iv, K_put, K_call, K_atm = self._fetch_skew_points(S, ivs, selected_strikes, actual_dte/365, r, q, mode, put_moneyness, call_moneyness, put_delta, call_delta)

            yield put_iv, call_iv, atm_iv, K_put, K_call, K_atm, close_date, S, ivs, strikes, params

    def atm_iv_model_test(self, dte: float, r: float = 0.04, q: float = 0.0, mode: str = "moneyness", put_moneyness: float = 0.1, call_moneyness: float = 0.1, put_delta: float = -0.25, call_delta: float = 0.25, **kwargs):
        atm_ivs, inst_vols, dates = [], [], []
        for put_iv, call_iv, atm_iv, K_put, K_call, K_atm, close_date, S, ivs, strikes, params in self._skew_data_generator(dte, r, q, mode, put_moneyness, call_moneyness, put_delta, call_delta, **kwargs):
            atm_ivs.append(atm_iv)
            inst_vols.append(params["inst_vol"])
            dates.append(close_date)

        return atm_ivs, inst_vols, dates

    def implied_inst_spot_vol_corr(self, dte, **kwargs):
        """
        This fetches us spot vol corr derived from some model
        """
        spots, dates, spot_vol_corrs = [], [], []
        for close_date, S, params in self._implied_parameters(dte, **kwargs):
            spots.append(S)
            dates.append(close_date)
            spot_vol_corrs.append(params["inst_spot_vol_corr"])

        return spots, dates, spot_vol_corrs
    
    def skew_curves(self, dte: float, r: float = .04, q: float = 0, **kwargs) -> (list[float], list[str], list[float]):

        spots, dates, skews = [], [], []
        for put_iv, call_iv, atm_iv, K_put, K_call, K_atm, close_date, S, ivs, strikes, params in self._skew_data_generator(dte, r, q, **kwargs):
            spots.append(S)
            dates.append(close_date)
            skews.append(ivs)

        return spots, dates, skews
    
    def implied_fixed_strike_spot_vol_beta(self, dte: float, r: float = 0.04, q: float = 0, weights: bool = False, put_moneyness: float = 0.1, call_moneyness: float = 0.1):
        """
        This method measures the skew curve using otm put and call ivs at a certain moneyness, then normalizes that by dividing by the strikes.
        This is a way to approximate the ATM skew slope using a finite difference type of model.
        This is from some content somewhere that Euan Sinclair put out.
        """
        implied_skews, dates = [], []
        for put_iv, call_iv, atm_iv, K_put, K_call, K_atm, close_date, S, ivs, strikes in self._skew_data_generator(dte, r, q, weights, mode="moneyness", put_moneyness=put_moneyness, call_moneyness=call_moneyness):
            dates.append(close_date)
            denom = (K_put - K_call) / S
            implied_skews.append((put_iv - call_iv) / denom)

        return implied_skews, dates

class RealizedSkew(RollingAnalytics):
    pass