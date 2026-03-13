import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm
from scipy.optimize import minimize  

from ..greeks.analytical_greeks import AnalyticalGamma, AnalyticalTheta, AnalyticalVanna, AnalyticalVolga, AnalyticalVega, AnalyticalSpeed, AnalyticalUltima, AnalyticalZomma, AnalyticalVolgaSpot
from py_op.utils.model_utils import GVVUtils

class InterpolationBaseClass:

    def skew(self):
        pass

    def implied_parameters(self):
        pass

class PolynomialNatenburg(InterpolationBaseClass):
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

    def skew(self, S: float, strikes: list[float], ivs: list[float] = None, dte: str = None, x_axis: str = 'log moneyness', a: float = None, b: float = None, c: float = None) -> list[float]:
        """
        parameters:
        a: atm iv, var swap, vix etc...
        b: skewness, (.25 delta call - .25 delta put), spot vol corr, etc...
        c: kurtosis, (.05 delta call - .05 delta put), vol vol, etc...

        We got rid of arguments a, b and c because we wanted this to be similar to gvv where we calculate the implied parameters and use those
        what I didnt realize is we want to be able to pass in parameters for atm iv, spot vol corr and vol or vol

        If we dont include ivs we have to use a, b, and c. We can use this with any range of strike grid we want ex np.linspace(500, 700, 300)
        """
        if any(v is None for v in (a, b, c)) and not all(v is None for v in (a, b, c)):
            raise ValueError("Either provide all of a, b, c or none of them.")
        
        if a is None and b is None and c is None:
            a, b, c = self.coefficents(S, strikes, ivs, dte, x_axis)

        x = self._x_axis(S, strikes, dte, x_axis)

        return a + b*x + c*x**2
    
    def coefficients(self, S, strikes, ivs, dte=None, x_axis='log moneyness'):
        assert len(ivs) == len(strikes), "Strikes must be the same len as ivs"
        x = self._x_axis(S, strikes, dte, x_axis)
        ivs = np.atleast_1d(ivs).reshape(-1, 1)
        
        X = np.column_stack([np.ones_like(x), x, x**2])
        beta, *_ = np.linalg.lstsq(X, ivs, rcond=None)
        a, b, c = beta
        return a, b, c
    

class GVV(GVVUtils, InterpolationBaseClass):
    """
    This version of GVV works really well, it gets us the correct coefficients and has two good fitting methods.
    In the future we should try to add some type of weighting mechanism to the linear regression, it will probably fit the skew polynomial better.
    We also want to probably combine both skew functions into one and have a paramter method: str as a argument to defne how we fit
    """

    def __init__(self, model_theta = AnalyticalTheta(), model_gamma = AnalyticalGamma(), model_vanna = AnalyticalVanna(), model_volga = AnalyticalVolga()) -> None:
        super().__init__(model_theta, model_gamma, model_vanna, model_volga)

    def coefficients(self, S: float, strikes: list[float], ivs: list[float], dte: float, weights: bool = False) -> (float, float, float):
        """
        This method calculates the coefficients of the model
        """
        strikes = np.asarray(strikes, dtype=float)
        ivs = np.asarray(ivs, dtype=float)

        if dte == 0:
            dte = .5/365

        theta = self.model_theta.calculate(S, strikes, dte, ivs, otype="call")
        gamma = 0.5*self.model_gamma.calculate(S, strikes, dte, ivs)*S**2
        vanna = self.model_vanna.calculate(S, strikes, dte, ivs)*ivs*S 
        volga = 0.5*self.model_volga.calculate(S, strikes, dte, ivs)*ivs**2

        X = np.column_stack((gamma, vanna, volga))
        y = -theta

        if weights == True:
            vega = AnalyticalVega().calculate(S, strikes, dte, ivs)
            w = 1 / vega
            X = X * w[:, None]
            y = y * w
            #print(f"w: {w}")

        coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        return coeffs 

    def skew(self, S: float, strikes: list[float], ivs_init: list[float], dte: float, weights: bool = False, method: str = 'bisection', interpolation_points: int = 200, b1 = None, b2 = None, b3 = None) -> list[float]:
        """
        Currently this uses a bisection method to fit the skew curve, we need to look into making it work better it tends to be very susceptible to outliers.
        !!! Look into this !!!
        Current methods: bisection, binary_search, secant 
        """

        coeffs = (b1, b2, b3)
        if any(p is None for p in coeffs) and not all(p is None for p in coeffs):
            raise ValueError("Either provide all of coefficients (b1, b2, b3) or none of them.")
        
        if b1 is None and b2 is None and b3 is None:
            coeffs = self.coefficients(S, strikes, ivs_init, dte, weights)

        method = "_".join(method.strip().lower().split())
        fn = getattr(self, f"get_iv_{method}")      

        if method == "polynomial":
            return fn(S, strikes, ivs_init, dte, coeffs)

        ivs_rooted, new_strikes = [], []
        for K in np.linspace(min(strikes), max(strikes), len(strikes)):

            iv = fn(S, K, dte, coeffs)
            if iv <= 0.0001:
                #ivs_rooted.append(ivs_init[i])
                continue
            else:
                ivs_rooted.append(iv)
                new_strikes.append(K)
        return np.array(ivs_rooted), np.array(new_strikes)
    
    def skew_polynomial_strike(self, F, strikes, ivs, dte, weights: bool = False):
        """
        This doesnt work very well need to work on it
        """
        ivs, strikes, r = np.array(ivs), np.array(strikes), 0

        vol_level, spot_vol, vol_vol = self.implied_parameters(F, strikes, ivs, dte, weights = False)

        d1 = (np.log(F / strikes) + (r + 0.5 * ivs ** 2) * dte) / (ivs * np.sqrt(dte))
        d2 = d1 - ivs * np.sqrt(dte)

        ivs = np.sqrt(vol_level**2 - 2*spot_vol*vol_vol*vol_level*np.sqrt(dte)*d2 + vol_vol**2 *d1*d2 * dte)
        return ivs, strikes

    def implied_parameters(self, S: float, strikes: list[float], ivs: list[float], dte: float, weights: bool = False, scale_spot_vol = False) -> (float, float, float):
        """
        When we use inverse vega weighting with out multiplying the denominator of spot vol it matches the parameters frido gets perfectly but sometimes it does go below -1 which doesnt make sense
        b1, b2, b3 are all perfect, so that means b2 = spot vol covariance
        We leave spot vol as divided by 2 for now but we might change in the future - we should compare these two different versions to realized spot vol
        """
        b1, b2, b3 = self.coefficients(S, strikes, ivs, dte, weights)
        vol_level, vol_vol = np.sqrt(b1), np.sqrt(b3)
        
        if scale_spot_vol == True:
            # this is the R implementation
            spot_vol_corr = b2 / (2*vol_level*vol_vol)

        else:
            # this is fridos version
            spot_vol_corr = b2 / (vol_level*vol_vol)

        #spot_vol_corr = np.clip(spot_vol_corr, -0.999, 0.999)
        return vol_level, spot_vol_corr, vol_vol
    
    def surface(self, strike_min, strike_max, dtes):
        pass

    def __repr__(self) -> str:
        return "GVV"

class GVV5:
    """
    This is Fridos GVV5 model, it works better than regular GVV so far
    """

    def __init__(self, model_theta = AnalyticalTheta(), model_gamma = AnalyticalGamma(), model_vanna = AnalyticalVanna(), model_volga = AnalyticalVolga()) -> None:
        self.model_theta = model_theta
        self.model_gamma = model_gamma
        self.model_vanna = model_vanna
        self.model_volga = model_volga

    def rho(self, F, K, gamma_param, m) -> list[float]:
        return np.tanh(gamma_param*np.log(K / F) - m)
    
    def implied_parameters(self, F, strikes, ivs, dte) -> list[float]:
        strikes, ivs = np.array(strikes), np.array(ivs)
        #print(f"median ivs: {np.median(ivs)}")
        #x0 = np.array([np.median(ivs), 0.5, .8, -.5, .5])
        x0 = np.array([.17, 0.5, .8, -.5, .5])

        lb = np.array([1e-4, 1e-6, -.5,  -10.0, -.5], dtype=float)
        ub = np.array([3.00,  4.0, 1.2,  10.0, 2.5], dtype=float)

        def residuals(x):
            sigma, omega, beta, gamma_param, m = x
            iv_model = self.skew(F, strikes, ivs, dte, sigma, omega, beta, gamma_param, m)
            return iv_model - ivs
        
        res = least_squares(residuals, x0=x0, bounds=(lb, ub), loss="linear")
        return res.x
    
    def skew(self, F, strikes, ivs, dte, sigma = None, omega = None, beta = None, gamma_param = None, m = None):

        params = (sigma, omega, beta, gamma_param, m)
        if any(p is None for p in params) and not all(p is None for p in params):
            raise ValueError("Either provide all of coefficients (b1, b2, b3) or none of them.")
        
        if sigma is None and omega is None and beta is None and gamma_param is None and m is None:
            sigma, omega, beta, gamma_param, m = self.implied_parameters(F, strikes, ivs, dte)

        gamma = self.model_gamma.calculate(F, strikes, dte, ivs)
        vanna = self.model_vanna.calculate(F, strikes, dte, ivs)
        volga = self.model_volga.calculate(F, strikes, dte, ivs)
        
        return np.sqrt(sigma ** 2 + 2*self.rho(F, strikes, gamma_param, m) * omega * sigma * ivs ** beta * vanna * (1 / (F*gamma)) + omega**2 * ivs ** (2*beta) * volga * (1 / (F**2 * gamma)))

class GVVPlus(GVVUtils):
    """
    The regular GVV works better so far but we have to work on this further. 
    I got the formula for dvolga/dspot from chat gpt so it is probably wrong
    """

    def __init__(self, model_theta = AnalyticalTheta(), model_gamma = AnalyticalGamma(), model_vanna = AnalyticalVanna(), model_volga = AnalyticalVolga(), model_speed = AnalyticalSpeed(), model_zomma = AnalyticalZomma(), model_ultima = AnalyticalUltima(), model_dvolga_dspot = AnalyticalVolgaSpot()):
        self.model_theta = model_theta
        self.model_gamma = model_gamma # should use AnalyticalGamma()
        self.model_vanna = model_vanna
        self.model_volga = model_volga
        self.model_speed = model_speed
        self.model_zomma = model_zomma
        self.model_ultima = model_ultima
        self.model_dvolga_dspot = model_dvolga_dspot

    def coefficients(self, S: float, strikes: list[float], ivs: list[float], dte: float, weights: bool = False) -> (float, float, float):
        """
        This method calculates the coefficients of the model
        """
        strikes = np.asarray(strikes, dtype=float)
        ivs = np.asarray(ivs, dtype=float)

        if dte == 0:
            dte = .5/365

        theta = self.model_theta.calculate(S, strikes, dte, ivs, otype="call")
        gamma = 0.5*self.model_gamma.calculate(S, strikes, dte, ivs)*S**2
        vanna = self.model_vanna.calculate(S, strikes, dte, ivs)*ivs*S 
        volga = 0.5*self.model_volga.calculate(S, strikes, dte, ivs)*ivs**2
        speed = (1/6)*self.model_speed.calculate(S, strikes, dte, ivs)*S**3
        zomma = (1/2)*self.model_zomma.calculate(S, strikes, dte, ivs)*S**2 * ivs
        dvolga_dspot = (1/2)*self.model_dvolga_dspot.calculate(S, strikes, dte, ivs) * ivs**2 * S
        ultima = (1/6)*self.model_ultima.calculate(S, strikes, dte, ivs)*ivs**3

        X = np.column_stack((gamma, vanna, volga, speed, dvolga_dspot, zomma, ultima))
        #X = np.column_stack((gamma, vanna, volga, speed, ultima))
        y = -theta

        if weights == True:
            vega = AnalyticalVega().calculate(S, strikes, dte, ivs)
            w = 1 / vega
            X = X * w[:, None]
            y = y * w
            #print(f"w: {w}")

        coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        return coeffs
    
    def skew(self, S: float, strikes: list[float], ivs_init: list[float], dte: float, weights: bool = False, method: str = 'bisection', interpolation_points: int = 200, b1 = None, b2 = None, b3 = None) -> list[float]:
        """
        This doesn't work very well
        """

        params = (b1, b2, b3)
        if any(p is None for p in params) and not all(p is None for p in params):
            raise ValueError("Either provide all of coefficients (b1, b2, b3) or none of them.")
        
        if b1 is None and b2 is None and b3 is None:
            coeffs = self.coefficients(S, strikes, ivs_init, dte, weights)

        method = "_".join(method.strip().lower().split())
        fn = getattr(self, f"get_iv_{method}")      

        ivs_rooted, new_strikes = [], []
        for K in np.linspace(min(strikes), max(strikes), len(strikes)):

            iv = fn(S, K, dte, coeffs)
            if iv <= 0.0001:
                #ivs_rooted.append(ivs_init[i])
                continue
            else:
                ivs_rooted.append(iv)
                new_strikes.append(K)
        return np.array(ivs_rooted), np.array(new_strikes)

    def implied_parameters(self, S: float, strikes: list[float], ivs: list[float], dte: float, weights: bool = False) -> (float, float, float):
        """
        Note for this vol level is perfect it matches atm iv for a constant maturity and vix over time so dont touch that.
        One day we will have to look at the scaling vol vol and spot vol, they are in the correct ranges but might need slight adjustments to the scalling
        """
        coefficients = self.coefficients(S, strikes, ivs, dte, weights)
        b1, b2, b3, *_ = coefficients
        vol_level, vol_vol = np.sqrt(b1), np.sqrt(b3)
        spot_vol_corr = b2 / (vol_level*vol_vol)
        return vol_level, spot_vol_corr, vol_vol
    
    def __repr__(self) -> str:
        return "GVVPlus"
    
class SVI:

    def implied_parameters(self, S: float, strikes: list[float], ivs: list[list[float]], dte: list[float], guess=[0.02, 0.5, -0.5, 0, 0.2], bounds=((None, None), (1e-8, 10), (-.999, .999), (None, None), (1e-8, 5)), method='SLSQP', tol=1e-3):

        def svi_error(S, strikes, market_ivs, dte, model, **kwargs):
            model_ivs = model(S, strikes, dte = dte, **kwargs)
            return np.sum((np.array(market_ivs) - np.array(model_ivs)) ** 2)

        opt_func = lambda x: svi_error(S, strikes, ivs, dte, self.skew, a = x[0], b = x[1], rho = x[2], m = x[3], sigma = x[4])
        result = minimize(opt_func, guess, bounds=bounds, method=method, tol=tol)
        params = result.x
        return params

    def skew(self, S: float, strikes: list[float], ivs: list[float] = None, dte: float = None, a: float = None, b: float = None, rho: float = None, m: float = None, sigma: float = None):
        """
        This is our function for calculating iv from the svi model, this function only does the calculation, no optmizing we just pass in parameters
        In calibration we have the optimizer for the parameters
        """
        if dte is None:
            raise ValueError("Must define dte")
        
        params = (a, b, rho, m, sigma)
        if any(p is None for p in params) and not all(p is None for p in params):
            raise ValueError("Either provide all of coefficients (b1, b2, b3) or none of them.")
        
        if a is None and b is None and rho is None and m is None and sigma is None:
            a, b, rho, m, sigma = self.implied_parameters(S, strikes, ivs, dte)

        k = np.log(strikes/S)
        total_var = a + b * (rho*(k - m) + np.sqrt((k - m)**2 + sigma**2))
        return np.sqrt(total_var/dte)

class IRV5(InterpolationBaseClass):
    pass

class VGVV(InterpolationBaseClass):
    pass

class SSVI(InterpolationBaseClass):
    pass

class ThreeToSmile(InterpolationBaseClass):
    pass

class VannaVolga(InterpolationBaseClass):
    pass


# This section for other ocde that is very useful

## Other GVV
### This GVV calculates the Gamma Vanna Volga cost framework using dollar greeks. It gives the same values as the other GVV but does not fit skew as well

class GVVDollarGreeks:
    """
    This version of GVV uses dollar greeks under blacks model (ie forward measure).
    This is the best version the old version using black scholes greeks in below, we might work on it in the future but this works really well.
    We need to make another method for fitting skew using something like a bisection/rootfinder instead of a polynomial
    """

    def coefficients(self, F, strikes, ivs, dte, r = 0):

        strikes, ivs = np.array(strikes), np.array(ivs)

        d1 = (np.log(F/strikes) + ((0.5*ivs*ivs)*dte))/(ivs * np.sqrt(dte))
        d2 = d1 - (ivs * np.sqrt(dte))

        bstheta = (-F*np.exp(-r*dte)*norm.pdf(d1)*ivs)/(2*np.sqrt(dte))
        dollar_gamma = (strikes/(ivs*np.sqrt(dte)))*norm.pdf((np.log(F/strikes)/(ivs*np.sqrt(dte))) - ivs*np.sqrt(dte)*0.5)
        dollar_vanna = (np.log(strikes/F) + 0.5*(ivs**2)*dte)*(F*norm.pdf(d1)/(ivs*np.sqrt(dte)))
        dollar_volga = F*norm.pdf(d1)*(-d1)*(-np.log(F/strikes) + 0.5*(ivs**2)*dte)

        X = np.column_stack((0.5*dollar_gamma, dollar_vanna, 0.5*dollar_volga))
        y = -bstheta

        coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

        return coeffs[0], coeffs[1], coeffs[2] 

    def implied_parameters(self, F, strikes, ivs, dte, r = 0):
        b1, b2, b3 = self.coefficients(F, strikes, ivs, dte, r = 0)

        return_var = b1
        spotvol_cov = b2
        implvol_var = b3

        return_vol = np.sqrt(abs(return_var))
        implvol_vol = np.sqrt(abs(implvol_var))
        implied_corr = spotvol_cov / (2*return_vol*implvol_vol)
        return return_vol, implied_corr, implvol_vol

    def skew(self, F, strikes, ivs, dte, r = 0):
        b1, b2, b3 = self.coefficients(F, strikes, ivs, dte, r = 0)

        return_var = b1
        spotvol_cov = b2
        implvol_var = b3

        return_vol = np.sqrt(abs(return_var))
        implvol_vol = np.sqrt(abs(implvol_var))
        implied_corr = spotvol_cov / (2*return_vol*implvol_vol)

        gvv_fit = np.sqrt(abs(return_var + 2*implied_corr*return_vol*implvol_vol*np.log(strikes/F) + implvol_var*(np.log(strikes/F)**2)))
        return gvv_fit, np.log(strikes/F)


# old code that we replaced

# GVV


class GVVOld(GVVUtils, InterpolationBaseClass):

    def __init__(self, model_theta = AnalyticalTheta(), model_gamma = AnalyticalGamma(), model_vanna = AnalyticalVanna(), model_volga = AnalyticalVolga()) -> None:
        super().__init__(model_theta, model_gamma, model_vanna, model_volga)

    def coefficients(self, S: float, strikes: list[float], ivs: list[float], dte: float,) -> (float, float, float):
        """
        This method calculates the coefficients of the model
        """
        strikes = np.asarray(strikes, dtype=float)
        ivs = np.asarray(ivs, dtype=float)

        if dte == 0:
            dte = .5/365

        theta = self.model_theta.calculate(S, strikes, dte, ivs, otype="call")
        gamma = 0.5*self.model_gamma.calculate(S, strikes, dte, ivs)*S**2
        vanna = self.model_vanna.calculate(S, strikes, dte, ivs)*ivs*S # multiplying by 10 seems to scale best
        volga = 0.5*self.model_volga.calculate(S, strikes, dte, ivs)*ivs**2 # multiplying by 2000 seems to scale best

        X = np.column_stack((gamma, vanna, volga))
        y = -theta

        coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        return coeffs[0], coeffs[1], coeffs[2] 

    def skew(self, S: float, strikes: list[float], ivs_init: list[float], dte: float, interpolation_points: int = 200, b1 = None, b2 = None, b3 = None) -> list[float]:
        """
        If method = "root finder"
         In this model we can either pass in a equal strike grid as our IV's in which case the code will calculate the coefficients for us,
         but strike and iv have to have the same length.
         Or we can pass in a longer lengthed strike grid than iv (ie we interpolate) but we have to calculate the coefficients first and pass them in (ie b1, b2, b3)

        If method = "least squares"
         If we want to interpolate between strikes we need to pass in a longer len iv array == len of strikes we want to interpolate over. We can do this using np.interp(arr_strikes, strikes, call_ivs)
         we can also pass in the regular strikes and ivs we get from OptionChain but they dont interpolate.
        """
        params = (b1, b2, b3)
        if any(p is None for p in params) and not all(p is None for p in params):
            raise ValueError("Either provide all of coefficients (b1, b2, b3) or none of them.")
        
        if b1 is None and b2 is None and b3 is None:
            b1, b2, b3 = self.coefficients(S, strikes, ivs_init, dte)
                    
        ivs_rooted, new_strikes = [], []
        for K in np.linspace(min(strikes), max(strikes), interpolation_points):
            iv = self.get_iv_bisection(S, K, dte, b1, b2, b3)
            if iv <= 0.06:
                #ivs_rooted.append(ivs_init[i])
                continue
            else:
                ivs_rooted.append(iv)
                new_strikes.append(K)
        return np.array(ivs_rooted), np.array(new_strikes)
        
    def implied_parameters(self, S: float, strikes: list[float], ivs: list[float], dte: float) -> (float, float, float):
        """
        Note for this vol level is perfect it matches atm iv for a constant maturity and vix over time so dont touch that.
        One day we will have to look at the scaling vol vol and spot vol, they are in the correct ranges but might need slight adjustments to the scalling
        """
        b1, b2, b3 = self.coefficients(S, strikes, ivs, dte)
        vol_level, vol_vol = np.sqrt(b1), np.sqrt(b3)
        spot_vol_corr = b2 / (2*vol_level*vol_vol)
        return vol_level, spot_vol_corr, vol_vol
    