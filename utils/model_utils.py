import numpy as np
from scipy.optimize import brentq


"""
This module is used purely to store extra functions or tools used in certain models
"""

class SABRUtils:

    @staticmethod
    def lognormal_vol(S: float, K: int, T: float, sigma_0: float, alpha: float, beta: float, rho: float, r: float = 0, q: float = 0):
        """
        this method is the log normal vol
        """
        if isinstance(K, (int, float)):
            if S == K:
                print(True)
                K += .001

        elif isinstance(K, (np.ndarray, list)):
            if np.any(K == S):
                idx = np.where(K == S)[0]
                K[idx] = K[idx] + .0001

        z = (alpha/sigma_0) * (S*K)**((1 - beta) / 2) * np.log(S/K)

        x = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))

        num_term1 = ((1 - beta) ** 2 / 24) * (sigma_0**2 / (S*K)**(1 - beta))
        num_term2 = (1/4)*(rho*beta*alpha*sigma_0) / (S*K)**((1 - beta)/2)
        num_term3 = ((2-3*rho**2) / 24)*alpha**2

        num = sigma_0*(1 + (num_term1 + num_term2 + num_term3)*T)

        denom_term1 = ((1 - beta)**2 / 24)*(np.log(S/K))**2
        denom_term2 = (((1 - beta)**4) / 1920) * (np.log(S/K))**4

        denom = (S*K)**((1 - beta)/2) * (1 + denom_term1 + denom_term2)

        return (num/denom)*(z/x)
    
    @staticmethod
    def atm_lognormal_vol(S: float, T: float, sigma_0: float, alpha: float, beta: float, rho: float, r: float = 0, q: float = 0):
        """
        This method finds the atm sigma but it does under estimate it.
        """
        num_term1 = ((1 - beta) ** 2 / 24) * (sigma_0**2 / (S)**(2 - 2*beta))
        num_term2 = (1/4)*(rho*beta*alpha*sigma_0) / (S)**(1 - beta)
        num_term3 = ((2-3*rho**2) / 24)*alpha**2

        num = sigma_0*(1 + (num_term1 + num_term2 + num_term3)*T)

        denom = S**(1-beta)

        return (num/denom)
    
    @staticmethod
    def normal_vol(S: float, K: float, T: float, sigma_0: float, alpha: float, beta: float, rho: float, r: float = 0, q: float = 0):
        """
        This method converts log normal vol to normal vol
        """
        if isinstance(K, (int, float)):
            if S == K:
                print(True)
                K += .001

        elif isinstance(K, (np.ndarray, list)):
            if np.any(K == S):
                idx = np.where(K == S)[0]
                K[idx] = K[idx] + .0001
                
        sigma_b_ = SABRUtils().lognormal_vol(S, K, T, sigma_0, alpha, beta, rho)
        sig_n = sigma_b_ * ((S - K) / np.log(S/K)) * (1 - ((sigma_b_**2 * T) / 24))
        return sig_n
    
    @staticmethod
    def asymptotic_normal_vol(S: float, K: float, T: float, sigma_0: float, alpha: float, beta: float, rho: float, r: float = 0, q: float = 0):
        """
        This method is the same from chris's book and what we learned in fixed income class
        Here we get a normal vol using the parameters intitial vol (sigma 0), alpha, beta and rho we then plug this normal
        vol into Bachelier to get a option price. Note Sigma 0 is not IV, vol or any other metric it is a specific parameter so dont plug
        in a vol value like 0.2 and expect to get a correct output you need to calibrate this.
        """
        # delta
        xi = (alpha / (sigma_0 * (1 - beta))) * (S**(1 - beta) - K**(1 - beta))
        num = np.sqrt(1 - 2*rho*xi + xi**2) + xi - rho
        denom = 1 - rho
        delta = np.log(num/denom)

        Smid = (S + K)/2
        CF = Smid**beta
        
        h = 0.0001
        c_prime = ((Smid + h)**beta - Smid**beta) / h
        c_prime_prime = ((Smid + h)**beta - 2*CF + (Smid - h)**beta) / (h**2)

        gamma1 = c_prime / CF
        gamma2 = c_prime_prime / CF

        epsilon = T*alpha**2

        asy_exp = (1+ ((2*gamma2 - gamma1**2)/24)*((sigma_0*CF)/alpha)**2 + ((rho*gamma1)/4)*((sigma_0*CF)/alpha) + (2-3*rho**2)/24)*epsilon
        return alpha * ((S-K)/delta)* asy_exp
    
class GVVUtils:

    def __init__(self, model_theta, model_gamma, model_vanna, model_volga):
        self.model_theta = model_theta
        self.model_gamma = model_gamma # should use AnalyticalGamma()
        self.model_vanna = model_vanna
        self.model_volga = model_volga

    def gvv_error_func_(self, S, K, dte, iv, coeffs):

        theta = self.model_theta.calculate(S, K, dte, iv)
        gamma = 0.5*self.model_gamma.calculate(S, K, dte, iv)*S**2
        vanna = self.model_vanna.calculate(S, K, dte, iv)*iv*S 
        volga = 0.5*self.model_volga.calculate(S, K, dte, iv)*iv**2
        greeks = [gamma, vanna, volga]

        if len(coeffs) == 7:
            speed = (1/6)*self.model_speed.calculate(S, K, dte, iv)*S**3
            zomma = (1/2)*self.model_zomma.calculate(S, K, dte, iv)*S**2 * iv
            dvolga_dspot = (1/2)*self.model_dvolga_dspot.calculate(S, K, dte, iv) * iv**2 * S
            ultima = (1/6)*self.model_ultima.calculate(S, K, dte, iv)*iv**3 
            greeks += [speed, zomma, dvolga_dspot, ultima]

        lhs = -theta
        #rhs = b1 * gamma + b2 * vanna + b3 * volga
        rhs = np.dot(coeffs, greeks)

        return lhs - rhs

    def get_iv_binary_search(self, S, K, dte, coeffs):
        ivs = np.linspace(0.05, 1, 1000)
        first, last = 0, len(ivs) - 1

        while first <= last:

            mid = (first + last) // 2

            val = self.gvv_error_func_(S, K, dte, ivs[mid], coeffs)

            if val < 0.000005 and val > 0:
                print(ivs[mid])
                break

            if val > 0:
                last = mid - 1

            else:
                first = mid + 1
        return ivs[mid]

    def get_iv_bisection(self, S, K, dte, coeffs):

        iv = 0.2
        lower_iv, upper_iv = 0.01, 2

        for i in range(1000):
            val = self.gvv_error_func_(S, K, dte, iv, coeffs)

            #if val < 0.0000005 and val > 0 or upper_iv - lower_iv < 0.001:
            if abs(val) < 5e-7 or (upper_iv - lower_iv) < 1e-3:

                return iv

            if val > 0:
                upper_iv = iv
            else:
                lower_iv = iv

            iv = (lower_iv + upper_iv) / 2

    def get_iv_secant(self, S, K, dte, b1, b2, b3, iv_lower = 0.05, iv_upper = 2, tol = 1e-8, max_iter=200) -> float:

        for i in range(max_iter):
            f_lower = self.gvv_error_func_(S, K, dte, iv_lower, b1, b2, b3)
            f_upper = self.gvv_error_func_(S, K, dte, iv_upper, b1, b2, b3)

            if abs(f_upper - f_lower) < 1e-14:
                raise ValueError("Denominator too small")
            
            new_iv = iv_upper - f_upper * (iv_upper - iv_lower) / (f_upper - f_lower)

            if abs(new_iv - iv_upper) < tol:
                return new_iv
            
            iv_lower, iv_upper = iv_upper, new_iv

        raise ValueError("Did not converge with max iter")
    
    def get_iv_polynomial(self, F, strikes, ivs, dte, coeffs) -> (list[float], list[float]):
        """
        This version fits decently well, the skew bisection fits a lot better, but this one is less succeptible to outliers.
        I am not sure if this is the closed form but we should try to see if it is and not make a version for the close form.      
        """
        return_var = coeffs[0]
        spotvol_cov = coeffs[1]
        implvol_var = coeffs[2]

        return_vol = np.sqrt(abs(return_var))
        implvol_vol = np.sqrt(abs(implvol_var))
        #implied_corr = spotvol_cov / (2*return_vol*implvol_vol)
        implied_corr = spotvol_cov / (return_vol*implvol_vol)

        gvv_fit = np.sqrt(abs(return_var + 2*implied_corr*return_vol*implvol_vol*np.log(strikes/F) + implvol_var*(np.log(strikes/F)**2)))
        return gvv_fit, np.log(strikes/F)