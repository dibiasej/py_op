from . import characteristic_functions as cf

import numpy as np
from scipy.integrate import quad
from functools import partial

"""
Look at the code in the github FNN
"""
def Q1(k, cf, right_lim):
    """
    P(X<k) - Probability to be in the money under the stock numeraire.
    cf: characteristic function
    right_lim: right limit of integration
    """

    def integrand(u):
        return np.real((np.exp(-u * k * 1j) / (u * 1j)) * cf(u - 1j) / cf(-1.0000000000001j))

    return 1 / 2 + 1 / np.pi * quad(integrand, 1e-15, right_lim, limit=2000)[0]


def Q2(k, cf, right_lim):
    """
    P(X<k) - Probability to be in the money under the money market numeraire
    cf: characteristic function
    right_lim: right limit of integration
    """

    def integrand(u):
        return np.real(np.exp(-u * k * 1j) / (u * 1j) * cf(u))

    return 1 / 2 + 1 / np.pi * quad(integrand, 1e-15, right_lim, limit=2000)[0]

class VarianceGammaFourierInversion:
    def call(self, S0: float, K: int, T: float, sigma: float, r: float = .04, theta: float = -.2, kappa: float =.4):
        """
        Price obtained by inversion of the characteristic function
        """
        k = np.log(K / S0)  # log moneyness
        w = -np.log(1 - theta * kappa - kappa / 2 * sigma**2) / kappa

        cf_VG_b = partial(
            cf.variance_gamma,
            t=T,
            mu=(r -w),
            theta=theta,
            sigma=sigma,
            kappa=kappa,
        )

        right_lim = 5000  # using np.inf may create warnings
        
        call = S0 * Q1(k, cf_VG_b, right_lim) - K * np.exp(-r * T) * Q2(k, cf_VG_b, right_lim)

        return call
        
    def put(self, S0, K, T, sigma, r = .04, theta = -.2, kappa=.4):
        """
        Price obtained by inversion of the characteristic function
        """
        k = np.log(K / S0)  # log moneyness
        w = -np.log(1 - theta * kappa - kappa / 2 * sigma**2) / kappa

        cf_VG_b = partial(
            cf.variance_gamma,
            t=T,
            mu=(r -w),
            theta=theta,
            sigma=sigma,
            kappa=kappa,
        )

        right_lim = 5000  # using np.inf may create warnings

        put = K * np.exp(-r * T) * (1 - Q2(k, cf_VG_b, right_lim)) - S0 * (1 - Q1(k, cf_VG_b, right_lim))

        return put

class HestonFourierInversion:

    def call(self, S: float, K: int, T: float, r: float = .02, q = 0, v0: float = .04, theta: float = .02, kappa: float = 2, sigma: float = .3, rho: float = -.5):

        """
        Price obtained by inversion of the characteristic function
        """
        k = np.log(K / S)  # log moneyness
        
        cf_H_b_good = partial(
            cf.heston,
            t=T,
            v0=v0,
            mu=r,
            theta=theta,
            sigma=sigma,
            kappa=kappa,
            rho=rho,
        )

        limit_max = 2000  # right limit in the integration

        call = S * Q1(k, cf_H_b_good, limit_max) - K * np.exp(-r * T) * Q2(k, cf_H_b_good, limit_max)
        return call

    def put(self, S: float, K: int, T: float, r: float = .02, q = 0, v0: float = .04, theta: float = .02, kappa: float = 2, sigma: float = .3, rho: float = -.5):

        """
        Price obtained by inversion of the characteristic function
        """
        k = np.log(K / S)  # log moneyness
        cf_H_b_good = partial(
            cf.heston,
            t=T,
            v0=v0,
            mu=r,
            theta=theta,
            sigma=sigma,
            kappa=kappa,
            rho=rho,
        )

        limit_max = 2000  # right limit in the integration

        put = K * np.exp(-r * T) * (1 - Q2(k, cf_H_b_good, limit_max)) - S * (1 - Q1(k, cf_H_b_good, limit_max))
        return put
    
class MertonJumpDiffusionFourierInversion:

    def call(self, S, K, T, r, sigma, lam, muJ, sigJ):
        """
        Price obtained by inversion of the characteristic function
        """
        k = np.log(K / S)  # log moneyness
        m = lam * (np.exp(muJ + (sigJ**2) / 2) - 1)  # coefficient m
        cf_Mert = partial(
            cf.merton_jump_diffusion,
            t=T,
            mu=(r - 0.5 * sigma**2 - m),
            sig=sigma,
            lam=lam,
            muJ=muJ,
            sigJ=sigJ,
        )

        call = S * Q1(k, cf_Mert, np.inf) - K * np.exp(-r * T) * Q2(k, cf_Mert, np.inf)  
        return call
        
    def put(self, S, K, T, r, sigma, lam, muJ, sigJ):
        """
        Price obtained by inversion of the characteristic function
        """
        k = np.log(K / S)  # log moneyness
        m = lam * (np.exp(muJ + (sigJ**2) / 2) - 1)  # coefficient m
        cf_Mert = partial(
            cf.merton_jump_diffusion,
            t=T,
            mu=(r - 0.5 * sigma**2 - m),
            sig=sigma,
            lam=lam,
            muJ=muJ,
            sigJ=sigJ,
        )

        put = K * np.exp(-r * T) * (1 - Q2(k, cf_Mert, np.inf)) - S * (1 - Q1(k, cf_Mert, np.inf))  # pricing function
        return put