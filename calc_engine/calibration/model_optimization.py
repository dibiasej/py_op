import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

from calc_engine.option_pricing import analytical_solutions as an
from calc_engine.option_pricing import FFT
from calc_engine.option_pricing import simulation as sim
from calc_engine.option_pricing.option_price_factory import OptionPriceFactory
from . import error_functions as ef

RISK_FREE_RATE = .04

class Optimizer:

    def __init__(self, model: str, error_function = None):
        self.model = model
        self.error_function = error_function

    def optimize(self):
        pass

class CEVOptimizer(Optimizer):
    def __init__(self, model: str, error_function = None):
        self.model = model
        self.error_function = error_function # ef.sum_of_squares_puts_and_calls

    def optimize(self, S: float, strikes: list[float], prices: list[float], T: float, r: float = 0, q: float = 0, guess=[0.4, 0.2], bounds=((1e-6, 10), (0.001, .99)), method='SLSQP', tol=1e-10):
        """
        Optimize the parameters of the CEV model using Scipy minimize.
        """
        opt_func = lambda x: self.error_function(S, strikes, prices, T, self.model, sigma=x[1], beta=x[0], r=r, q = q) 
        result = minimize(opt_func, guess, bounds=bounds, method=method, tol=tol)
        params = result.x 
        return params
    
class HestonOptimizer(Optimizer):

    def __init__(self, model, error_function = None):
        self.model = model
        self.error_function = error_function #ef.normalized_root_mean_squared_error_matrix
    
    def optimize(self, S: float, strikes: list[float], prices: list[list[float]], T: list[float], r: float = 0, q: float = 0, guess=[0.3, .06, .05, 2.5, -.8], bounds=((1e-2, None), (1e-3, None), (1e-3, None), (1e-3, None), (-.95, .95)), method='SLSQP', tol=1e-3):
        """
        Optimize the parameters of the CEV model using Scipy minimize.
        bounds=((1e-6, None), (1e-6, None), (1e-6, None), (-.95, .95))
        bounds=((1e-2, 1), (1e-3, 0.1), (1e-3, 0.1), (1e-3, 5), (-.95, .95))
        bounds=((1e-6, 1), (1e-6, .1), (1e-6, 5), (-.95, .95))
        """
        #opt_func = lambda x: self.error_function(S, strikes, prices, T, self.model, sigma=x[0], v0 = x[1], theta = x[2], kappa = x[3], rho = x[4], r=r, q = q) 

        def opt_func(x):
            sigma, v0, theta, kappa, rho = x

            # Feller condition: 2 * kappa * theta > sigma^2
            # Penalize Feller condition violations
            if 2 * kappa * theta <= sigma**2:
                return np.inf

            return self.error_function(S, strikes, prices, T, self.model, sigma=sigma, v0=v0, theta=theta, kappa=kappa, rho=rho, r=r, q=q)
        
        result = minimize(opt_func, guess, bounds=bounds, method=method, tol=tol)
        print(result)
        params = result.x 
        return params
    
class VarianceGammaOptimizer(Optimizer):
    pass

class MertonJumpDiffusionOptimizer(Optimizer):
    pass
    
class SABROptimizer:
    """
    This is a very simple way to optimize sabr, later we will make this faster by using the method from the SABR paper we read.
    """
    
    def __init__(self, model: str, error_function = None):
        self.model = model
        self.error_function = error_function
    
    def optimize(self, S: float, strikes: list[float], ivs: list[float], T: float, r: float = 0.04, q: float = 0, beta: float = .5, guess=[.3, .5, -.2], bounds=((0.001, None), (0.001, None), (-0.95, 0.95)), method='SLSQP', tol=1e-10):
        print(f"Model: {self.model}")
        opt_func = lambda x: self.error_function(S, strikes, ivs, T, self.model, sigma_0=x[0], alpha=x[1], beta = beta, rho=x[2])
        result = minimize(opt_func, guess, bounds=bounds, method=method, tol=tol)
        params = result.x
        return params

class rBergomiOptimizer(Optimizer):
    
    def __init__(self, model: str, error_function = None):
        self.model = model
        self.error_function = error_function #ef.sum_of_squares_puts_and_calls
    
    def optimize(self, S: float, strikes: np.ndarray, prices: np.ndarray, T: float, r = RISK_FREE_RATE, guess=[.055, -.4, -.9, 1.9], bounds=((.01, .14), (-.49, .49), (-.9, .9), (.01, 5)), method='SLSQP', tol=1e-10):
        opt_func = lambda x: self.error_function(S, strikes, prices, T, self.model, xi = x[0], a=x[1], rho=x[2], eta = x[3])
        result = minimize(opt_func, guess, bounds=bounds, method=method, tol=tol)
        params = result.x
        return params
    
class OptimizerFactory:

    @staticmethod
    def create_optimizer(model_name: str, model, error_function):
        model_name = model_name.lower()
        print(f"model name {model_name}")
        if model_name == 'cev':
            return CEVOptimizer(model, error_function)
        elif model_name == 'heston':
            return HestonOptimizer(model, error_function)
        elif model_name == 'sabr':
            return SABROptimizer(model, error_function)
        elif model_name in ('rbergomi', 'rough bergomi'):
            return rBergomiOptimizer(model, error_function)
        else:
            raise ValueError("model must be specified")