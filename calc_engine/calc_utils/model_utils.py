import numpy as np


"""
This module is used purely to store extra functions or tools used in certain models
"""

class SABRUtils:

    @staticmethod
    def lognormal_vol(S: float, K: int, T: float, sigma_0: float, alpha: float, beta: float, rho: float):
        """
        this method is the log normal vol
        """
        if S == K:
            K += .000001
            print(f"Strike {K}")
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
    def atm_lognormal_vol(S: float, T: float, sigma_0: float, alpha: float, beta: float, rho: float):
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
    def normal_vol(S: float, K: float, T: float, sigma_0: float, alpha: float, beta: float, rho: float):
        """
        This method converts log normal vol to normal vol
        """
        sigma_b_ = SABRUtils().lognormal_vol(S, K, T, sigma_0, alpha, beta, rho)
        sig_n = sigma_b_ * ((S - K) / np.log(S/K)) * (1 - ((sigma_b_**2 * T) / 24))
        return sig_n
    
    @staticmethod
    def asymptotic_normal_vol(S: float, K: float, T: float, sigma_0: float, alpha: float, beta: float, rho: float):
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