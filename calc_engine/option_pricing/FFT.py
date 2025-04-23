import numpy as np
from abc import ABC, abstractmethod
from functools import partial
from scipy.fftpack import ifft
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

from . import characteristic_functions as cf

class FFT_Template(ABC):

    @abstractmethod
    def call(self):
        pass

    @abstractmethod
    def put(self):
        pass

    def fft_Lewis(self, S: float, strikes: list[int], r: float, T: float, characteristic_function, interp: str="cubic"):

        N = 2**15  # FFT more efficient for N power of 2
        B = 500  # integration limit
        dx = B / N
        x = np.arange(N) * dx  # the final value B is excluded

        weight = np.arange(N)  # Simpson weights
        weight = 3 + (-1) ** (weight + 1)
        weight[0] = 1
        weight[N - 1] = 1

        dk = 2 * np.pi / B
        b = N * dk / 2
        ks = -b + dk * np.arange(N)

        integrand = np.exp(-1j * b * np.arange(N) * dx) * characteristic_function(x - 0.5j) * 1 / (x**2 + 0.25) * weight * dx / 3
        integral_value = np.real(ifft(integrand) * N)
        
        if interp == "linear":
            spline_lin = interp1d(ks, integral_value, kind="linear")
            prices = S - np.sqrt(S * strikes) * np.exp(-r * T) / np.pi * spline_lin(np.log(S / strikes))
        elif interp == "cubic":
            spline_cub = interp1d(ks, integral_value, kind="cubic")
            prices = S - np.sqrt(S * strikes) * np.exp(-r * T) / np.pi * spline_cub(np.log(S / strikes))
        
        return prices

class HestonFFT(FFT_Template):

    def __init__(self, characteristic_function = cf.heston_schoutens) -> None:
        self.characteristic_function = characteristic_function

    def call(self, S: float, strikes: list[int], T: float, K: int = None, r: float = .02, q:float = 0, v0: float = .04, theta: float = .02, kappa: float = 2, sigma: float = .3, rho: float = -.5, interp = 'cubic'):
        """
        FFT method. It returns a vector of prices.
        K is an array of strikes
        """
        strikes = np.array(strikes)
        cf_H_b_good = partial(self.characteristic_function, t=T, v0=v0, mu=r, theta=theta, sigma=sigma, kappa=kappa, rho=rho)

        prices = self.fft_Lewis(S, strikes, r, T, cf_H_b_good, interp=interp)

        if K == None:
            return prices
        else:
            index = np.where(strikes == K)[0][0]
            return prices[index]
        
    def put(self, S: float, strikes: list[int], T: float, K: int = None, r: float = .02, q = 0, v0: float = .04, theta: float = .02, kappa: float = 2, sigma: float = .3, rho: float = -.5, interp = 'cubic'):
        """
        FFT method. It returns a vector of prices.
        K is an array of strikes
        """
        strikes = np.array(strikes)
        cf_H_b_good = partial(self.characteristic_function, t=T, v0=v0, mu=r, theta=theta, sigma=sigma, kappa=kappa, rho=rho)

        prices = self.fft_Lewis(S, strikes, r, T, cf_H_b_good, interp=interp) - S + strikes * np.exp(-r * T)

        if K == None:
            return prices
        else:
            index = np.where(strikes == K)[0][0]
            return prices[index]

class MertonJumpDiffusionFFT(FFT_Template):

    def __init__(self, characteristic_function = cf.merton_jump_diffusion) -> None:
        self.characteristic_function = characteristic_function

    def call(self, S: float, strikes: list[int], T: float, K: int = None, r = .02, mu=1, sig=2, lam=0.8, muJ=0, sigJ=0.5, interp = 'cubic'):
        """
        FFT method. It returns a vector of prices.
        K is an array of strikes
        """
        strikes = np.array(strikes)
        m = lam * (np.exp(muJ + (sigJ**2) / 2) - 1)  # coefficient m
        cf_Mert = partial(self.characteristic_function, t=T, mu=(r - 0.5 * sig**2 - m), sig=sig, lam=lam, muJ=muJ, sigJ=sigJ)

        prices = self.fft_Lewis(S, strikes, r, T, cf_Mert, interp=interp)

        if K == None:
            return prices
        else:
            index = np.where(strikes == K)[0][0]
            return prices[index]
        
    def put(self, S: float, strikes: list[int], T: float, K: int = None, r = .02, mu=1, sig=2, lam=0.8, muJ=0, sigJ=0.5, interp = 'cubic'):
        """
        FFT method. It returns a vector of prices.
        K is an array of strikes
        """
        strikes = np.array(strikes)
        m = lam * (np.exp(muJ + (sigJ**2) / 2) - 1)  # coefficient m
        cf_Mert = partial(self.characteristic_function, t=T, mu=(r - 0.5 * sig**2 - m), sig=sig, lam=lam, muJ=muJ, sigJ=sigJ)

        prices = self.fft_Lewis(S, strikes, r, T, cf_Mert, interp=interp) - S + strikes * np.exp(-r * T)

        if K == None:
            return prices
        else:
            index = np.where(strikes == K)[0][0]
            return prices[index]
    
class VarianceGammaFFT(FFT_Template):

    def __init__(self, characteristic_function = cf.variance_gamma) -> None:
        self.characteristic_function = characteristic_function

    def call(self, S: float, strikes: list[int], T: float, K: int = None, r: float = .02, w: float = 0, theta: float = -.1, sigma: float = .2, kappa: float = .1, interp: str = 'cubic'):
        """
        FFT method. It returns a vector of prices.
        K is an array of strikes
        """
        strikes = np.array(strikes)
        cf_VG_b = partial(self.characteristic_function, t=T, mu=(r - w), theta=theta, sigma=sigma, kappa=kappa)

        prices = self.fft_Lewis(S, strikes, r, T, cf_VG_b, interp=interp)

        if K == None:
            return prices
        else:
            index = np.where(strikes == K)[0][0]
            return prices[index]

    def put(self, S: float, strikes: list[int], T: float, K: int = None, r: float = .02, w: float = 0, theta: float = -.1, sigma: float = .2, kappa: float = .1, interp: str = 'cubic'):
        """
        FFT method. It returns a vector of prices.
        K is an array of strikes
        """
        strikes = np.array(strikes)
        cf_VG_b = partial(self.characteristic_function, t=T, mu=(r - w), theta=theta, sigma=sigma, kappa=kappa)

        prices = self.fft_Lewis(S, strikes, r, T, cf_VG_b, interp="cubic") - S + strikes * np.exp(-r * T)

        if K == None:
            return prices
        else:
            index = np.where(strikes == K)[0][0]
            return prices[index]