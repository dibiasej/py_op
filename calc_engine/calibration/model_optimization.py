import numpy as np
from scipy.optimize import minimize

from calc_engine.option_pricing import analytical_solutions as an
from calc_engine.option_pricing import FFT
from calc_engine.option_pricing import simulation as sim
from . import error_functions as ef

class Optimizer:

    def __init__(self, error_function = ef.sum_of_squares, model_method = None):
        self.error_function = error_function
        self.model_method = model_method
    def optimize(self):
        pass

class CEVOptimizer(Optimizer):
    def __init__(self, error_function = ef.sum_of_squares, model_method = an.CEVAnalytical()):
        self.error_function = error_function
        self.model_method = model_method

    def optimize(self, S: float, strikes: list[float], prices: list[float], T: float, r: float = 0, q: float = 0, guess=[0.4, 0.2], bounds=((1e-6, 10), (0.001, .99)), method='SLSQP', tol=1e-10):
        """
        Optimize the parameters of the CEV model using Scipy minimize.
        """
        opt_func = lambda x: self.error_function(S, strikes, prices, T, self.model_method, sigma=x[1], beta=x[0], r=r, q = q) 
        result = minimize(opt_func, guess, bounds=bounds, method=method, tol=tol)
        params = result.x 
        return params
    
class HestonOptimizer(Optimizer):
    """
    We could not get this to properly optimize in the future I will have to vectorize the price surface
    """
    def __init__(self, error_function = ef.sum_of_squares_heston_skew, model_method = FFT.HestonFFT()):
        self.error_function = error_function
        self.model_method = model_method

    def manual_optimization(self, S: float, strikes: list[float], prices: list[list[float]], T: list[float], v0: float, theta: float, r: float = 0, q: float = 0, otype = 'call'):
        """
        This function is not done but we saved the code here for future updating
        We probably shouldnt run this because it will take like an hour and nobody got time for that just use scipy
        Turn theta into sigma(atm) for specific T, or use k / dte where k =2.75
        """
        kappa_list = [2, 2.5, 3.5, 5]
        rho_list = [-1, -.8, -.6, -.5]
        sigma_list = [.2, .3, .5, .8]
        sse_hest = []
        for k in kappa_list:
            for rh in rho_list:
                for sig in sigma_list:

                    # implement feller condition
                    if 2*k*theta - sig**2 < 0:
                        continue

                    erf_sum_heston = self.error_function(S, strikes, prices, np.array(T) / 252, self.model_method, v0 = v0, kappa = k, theta = theta, rho = rh, sigma = sig)
                    sse_hest.append((erf_sum_heston, k, rh, sig))

        min_tuple = min(sse_hest, key=lambda x: x[0])

        return min_tuple
    
    def optimize(self, S: float, strikes: list[float], prices: list[list[float]], T: list[float], r: float = 0, q: float = 0, otype = 'call', guess=[0.3, .06, .05, 2.5, -.8], bounds=((1e-2, 1), (1e-3, 0.1), (1e-3, 0.1), (1e-3, 5), (-.95, .95)), method='SLSQP', tol=1e-10):
        """
        Optimize the parameters of the CEV model using Scipy minimize.
        bounds=((1e-6, None), (1e-6, None), (1e-6, None), (-.95, .95))
        bounds=((1e-2, 1), (1e-3, 0.1), (1e-3, 0.1), (1e-3, 5), (-.95, .95))
        bounds=((1e-6, 1), (1e-6, .1), (1e-6, 5), (-.95, .95))
        """
        opt_func = lambda x: self.error_function(S, strikes, prices, T, self.model_method, otype = otype, sigma=x[0], v0 = x[1], theta = x[2], kappa = x[3], rho = x[4], r=r, q = q) 
        result = minimize(opt_func, guess, bounds=bounds, method=method, tol=tol)
        print(result)
        params = result.x 
        return params
    
class VarianceGammaOptimizer(Optimizer):
    pass

class MertonJumpDiffusionOptimizer(Optimizer):
    pass
    
class SABROptimizer(Optimizer):
    """
    This is a very simple way to optimize sabr, later we will make this faster by using the method from the SABR paper we read.
    """
    
    def __init__(self, error_function = ef.sse_log_ivs, model_method = an.SABRAnalytical()):
        self.error_function = error_function
        self.model_method = model_method
        #bounds=((0.1, 0.5), (0.01, .99), (-0.99, .99), (0.1, 1))

    def manual_optimizer(self, S: float, strikes: list[float], ivs: list[float], T: float, beta: float=.99):
        alpha_list = np.linspace(.01, 5, 30) #[.1, .2, .4, .6, .8] 
        rho_list = np.linspace(-.99, .99, 30) #[-.5, -.2, .2, .5, .8]
        sigma_list = np.linspace(.01, .8, 30) #[.1, .25, .4, .5, .65]

        sse_list = []
        for i in range(len(alpha_list)):
            for j in range(len(sigma_list)):
                for k in range(len(rho_list)):
                    sse = self.error_function(S, strikes, ivs, T, self.model_method, sigma_0=sigma_list[j], alpha=alpha_list[i], beta=beta, rho=rho_list[k])
                    sse_list.append((sse, sigma_list[j], alpha_list[i], rho_list[k]))

        min_params = min(sse_list)

        return [min_params[1], min_params[2], min_params[3]]
    
    def optimize(self, S: float, strikes: list[float], ivs: list[float], T: float, beta: float = .99, guess=[.3, .5, -.2], bounds=((0.001, None), (0.001, None), (-0.95, 0.95)), method='SLSQP', tol=1e-10):
        opt_func = lambda x: self.error_function(S, strikes, ivs, T, self.model_method, sigma_0=x[0], alpha=x[1], beta = beta, rho=x[2])
        result = minimize(opt_func, guess, bounds=bounds, method=method, tol=tol)
        params = result.x
        return params
    
class rBergomiOptimizer(Optimizer):
    
    def __init__(self, error_function = ef.sum_of_squares_puts_and_calls, model_method = sim.rBergomiSimulation()):
        self.error_function = error_function
        self.model_method = model_method
        #bounds=((0.1, 0.5), (0.01, .99), (-0.99, .99), (0.1, 1))

    def manual_optimizer(self, S: float, strikes: list[float], prices: list[float], T: float, xi: float):
        a_list = np.linspace(-.49, .49, 30) #[.1, .2, .4, .6, .8] 
        rho_list = np.linspace(-.99, .99, 30) #[-.5, -.2, .2, .5, .8]
        eta_list = np.linspace(.01, 7, 30) #[.1, .25, .4, .5, .65]

        sse_list = []
        for i in range(len(a_list)):
            for j in range(len(rho_list)):
                for k in range(len(eta_list)):
                    sse = self.error_function(S, strikes, prices, T, self.model_method, xi = xi, a=a_list[j], rho=rho_list[i], eta=eta_list[k])
                    sse_list.append((sse, a_list[j], rho_list[i], eta_list[k]))

        min_params = min(sse_list)

        return [min_params[1], min_params[2], min_params[3]]
    
    def optimize(self, S: float, strikes: np.ndarray, prices: np.ndarray, T: float, xi: float, guess=[-.2, -.9, 1.9], bounds=((-.49, .49), (-.99, .99), (.01, 7)), method='SLSQP', tol=1e-10):
        opt_func = lambda x: self.error_function(S, strikes, prices, T, self.model_method, xi = xi, a=x[0], rho=x[1], eta = x[2])
        result = minimize(opt_func, guess, bounds=bounds, method=method, tol=tol)
        params = result.x
        return params