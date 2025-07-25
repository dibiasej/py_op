from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm

#from data.option_data import OptionNode, TODAY

"""
Note: Because we imported data in this fashion we cannot directly call it in this module in a if __name__ == __main__ type of section.
We have to call/test this module in the main.py file in the root of our directory or another module outside of our directory


Note: We are going to try to use the Adapter pattern ofr calculating the greeks, the Adapter will be call OptionNodeGreek and it will take a Greek class (ie Delta) as input
     - This is also almost identical to the Bridge design pattern as well
     - (3/09/25) - We will make another module or in our visualization code we will create an adapter/bridge pattern that will calculate the greeks
                    by taking in a OptionNode

Note: We might turn all these classes into functions if there is no more functionality to add

!!! We still need to calculate Veta and Zomma!!!
"""
"""
class AnalyticalGreek(ABC):

    @abstractmethod
    def calculate(self) -> float:
        pass"""

class AnalyticalDelta:

    @staticmethod
    def calculate(S: int, K: int, T: float, sigma: float, r: float = 0, q: float = 0, otype: str = "call") -> float:

        if T > 1:
            T /= 365

        d1: float = (np.log(S/K) + (r - q + ((sigma**2)/2))*T) / (sigma * np.sqrt(T))
        
        if otype == "call":
            call_delta: float = np.exp(-q * T) * norm.cdf(d1)
            return call_delta

        elif otype == "put":
            put_delta: float = -np.exp(-q*T) * norm.cdf(-d1)
            return put_delta
    
    def __repr__(self):
        return f"Delta"
    
class AnalyticalTheta:
    @staticmethod
    def calculate(S: float, K: float, T: float, sigma: float, r: float = 0.0, q: float = 0.0, otype: str = "call") -> float:
        if T > 1:
            T /= 365  # Assume user passed T in trading days

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if otype == "call":
            theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                     - r * K * np.exp(-r * T) * norm.cdf(d2) 
                     + q * S * np.exp(-q * T) * norm.cdf(d1))
        elif otype == "put":
            theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                     + r * K * np.exp(-r * T) * norm.cdf(-d2) 
                     - q * S * np.exp(-q * T) * norm.cdf(-d1))
        else:
            raise ValueError("otype must be 'call' or 'put'")

        return theta /100
    
class AnalyticalGamma:

    # Note the Otype parameter does not matter for this
    @staticmethod
    def calculate(S: int, K: int, T: float, sigma: float, r: float = 0, q: float = 0, otype: str = "call") -> float:

        if T > 1:
            T /= 365

        d1: float = (np.log(S/K) + (r - q + ((sigma**2)/2)*T)) / (sigma * np.sqrt(T))
        gamma: float = (np.exp(-q*T) * norm.pdf(d1)) / (sigma * S * np.sqrt(T))
        return gamma
     
    def __repr__(self) -> str:
        return "Gamma"
    
class AnalyticalVega:

    # Note the Otype parameter does not matter for this
    @staticmethod
    def calculate(S: int, K: int, T: float, sigma: float, r: float = 0, q: float = 0, otype: str = "call") -> float:

        if T > 1:
            T /= 365

        d1 = (np.log(S/K) + (r - q + ((sigma**2)/2)) * T) / (sigma*np.sqrt(T))
        vega: float = S*np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T)
        return vega
    
    def __repr__(self) -> str:
        return f"Vega"
    
class AnalyticalVanna:

    def calculate(self, S: int, K: int, T: float, sigma: float, r: float = 0, q: float = 0, otype: str = "call") -> float:

        if T > 1:
            T /= 365

        d1: float = (np.log(S/K) + (r - q + ((sigma**2)/2))*T) / (sigma * np.sqrt(T))
        d2: float = d1 - sigma*np.sqrt(T)
        vanna: float =  - np.exp(-q * T) * (1 / np.sqrt(2 * np.pi)) * np.exp(-((d1 ** 2)/2)) * (d2 / sigma)
        return vanna
    
class AnalyticalVolga:

    def calculate(self, S: int, K: int, T: float, sigma: float, r: float = 0, q: float = 0, otype: str = "call") -> float:

        if T > 1:
            T /= 365

        d1: float = (np.log(S/K) + (r - q + ((sigma**2)/2))*T) / (sigma * np.sqrt(T))
        d2: float = d1 - sigma*np.sqrt(T)
        volga: float = np.sqrt(T) * norm.pdf(d1) * ((d1*d2)/sigma)
        return volga
    
class AnalyticalCharm:

    def calculate(self, S: int, K: int, T: float, sigma: float, r: float = 0, q: float = 0, otype: str = "call") -> float:

        if T > 1:
            T /= 365

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if otype == "call":
            term1 = q * np.exp(-q * T) * norm.cdf(d1)
            term2 = np.exp(-q * T) * norm.pdf(d1) * ((2 * (r - q) * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T)))
            charm = term1 - term2
            return charm
        
        elif otype == "put":
            term1 = -q * np.exp(-q * T) * norm.cdf(-d1)
            term2 = np.exp(-q * T) * norm.pdf(d1) * ((2 * (r - q) * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T)))
            charm = term1 - term2
            return charm
        
class AnalyticalSpeed:

    def calculate(self, S: int, K: int, T: float, sigma: float, r: float = 0, q: float = 0, otype: str = "call") -> float:

        if T > 1:
            T /= 365

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        speed = -gamma / S * (1 + d1 / (sigma * np.sqrt(T)))
        return speed
    
class AnalyticalColor:

    def calculate(self, S: int, K: int, T: float, sigma: float, r: float = 0, q: float = 0, otype: str = "call") -> float:

        if T > 1:
            T /= 365

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        color = -np.exp(-q * T) * (norm.pdf(d1) / (2 * S * T * sigma * np.sqrt(T))) * (2 * q * T + 1 + ((2*(r - q) * T - d2 * sigma * np.sqrt(T)) / (sigma * np.sqrt(T))) * d1)
        return -color
    
class AnalyticalRho:

    def calculate(self, S: int, K: int, T: float, sigma: float, r: float = 0, q: float = 0, otype: str = "call") -> float:

        if T > 1:
            T /= 252

        d1: float = (np.log(S/K) + (r - q + ((sigma**2)/2))*T) / (sigma * np.sqrt(T))
        d2: float = d1 - sigma*np.sqrt(T)

        if otype == "call":
            rho: float = K * T * np.exp(-r*T) * norm.cdf(d2)
            return rho
        
        elif otype == "put":
            rho = -K * T * np.exp(-r*T) * norm.cdf(-d2)
            return rho