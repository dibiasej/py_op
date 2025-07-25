import numpy as np
from scipy.optimize import least_squares

from ..greeks.analytical_greeks import AnalyticalGamma, AnalyticalTheta, AnalyticalVanna, AnalyticalVolga
from utils.model_utils import GVVUtils

class PointEstimateBaseClass:

    def skew(self):
        pass

    def implied_parameters(self):
        pass

class PolynomialNatenburg(PointEstimateBaseClass):
    """
    This class is based of the polynomial skew fitting model from natenburg.
    We can calculate skew using the skew method and pass in spot strikes, atm iv, skewness and kurtosis.
    We can fit atm iv, implied spot vol corr and implied vol vol using the fit 
    Helper function _x_axis for transforming the parameterization of the skew curve
    """
    def _x_axis(self, S, strikes, dte=None, x_axis='log moneyness'):
        """
        Helper function for x axis
        """
        strikes = np.array(strikes)
        if x_axis == "log moneyness":
            x = np.log(strikes/S)
        
        elif x_axis == "square root time":
            x = np.log(strikes/S) / np.sqrt(dte)
        
        return x
    
    def _y_axis(self):
        pass

    def skew(self, S, strikes, a, b, c, dte=None, x_axis='log moneyness'):
        """
        parameters:
        a: atm iv, var swap, vix etc...
        b: skewness, (.25 delta call - .25 delta put), spot vol corr, etc...
        c: kurtosis, (.05 delta call - .05 delta put), vol vol, etc...
        """
        x = self._x_axis(S, strikes, dte, x_axis)

        return a + b*x + c*x**2
    
    def implied_parameters(self, S, strikes, ivs, dte=None, x_axis='log moneyness'):
        assert len(ivs) == len(strikes), "Strikes must be the same len as ivs"
        x = self._x_axis(S, strikes, dte, x_axis)
        ivs = np.atleast_1d(ivs).reshape(-1, 1)
        
        X = np.column_stack([np.ones_like(x), x, x**2])
        beta, *_ = np.linalg.lstsq(X, ivs, rcond=None)
        a, b, c = beta
        return a, b, c
    
class GVV(GVVUtils, PointEstimateBaseClass):

    def __init__(self, model_theta = AnalyticalTheta(), model_gamma = AnalyticalGamma(), model_vanna = AnalyticalVanna(), model_volga = AnalyticalVolga()) -> None:
        super().__init__(model_theta, model_gamma, model_vanna, model_volga)

    def calculate(self, S: float, strikes: list[float], dte: float, ivs: list[float]) -> (float, float, float):

        theta = self.model_theta.calculate(S, strikes, dte, ivs)
        gamma = self.model_gamma.calculate(S, strikes, dte, ivs)
        vanna = self.model_vanna.calculate(S, strikes, dte, ivs)
        volga = self.model_volga.calculate(S, strikes, dte, ivs)

        lhs, rhs = np.array([gamma*S, vanna, volga*ivs]).T, -theta
        coeffs, residuals, rank, s = np.linalg.lstsq(lhs, rhs, rcond=None)
        return coeffs[0], coeffs[1], coeffs[2]

    def skew(self, S: float, strikes: list[float], dte: float, ivs_init: list[float], method: str = 'root finder') -> list[float]:

        if method == 'least squares':
            res = least_squares(
                self.gvv_error_func_least_squares, ivs_init,
                args=(S, strikes, dte),
                bounds=(0.01, 2.0),
                xtol=1e-8, ftol=1e-8
            )
            return res.x
        
        if method == 'root finder':
            b1, b2, b3 = self.calculate(S, strikes, dte, ivs_init)
            ivs_rooted = []
            for i, K in enumerate(strikes):
                iv = self.get_iv_bisection(S, K, dte, b1, b2, b3)
                if iv <= 0.06:
                    ivs_rooted.append(ivs_init[i])
                else:
                    ivs_rooted.append(iv)
            return np.array(ivs_rooted)
        
    def implied_parameters(self, S: float, strikes: list[float], dte: float, ivs: list[float]) -> (float, float, float):
        b1, b2, b3 = self.calculate(S, strikes, dte, ivs)
        atm_iv, vol_vol = np.sqrt(b1), np.sqrt(b3)
        spot_vol_cov = b2 / (atm_iv*vol_vol)
        return atm_iv, spot_vol_cov, vol_vol
    
class SVI(PointEstimateBaseClass):
    pass

class SSVI(PointEstimateBaseClass):
    pass