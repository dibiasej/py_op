import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from typing import List
from variables import Variables

class Formulas(Variables):
    def __init__(self, S: float, K: float, T: float, sigma: float, r: float, q: float):
        super().__init__(S, K, T, sigma, r, q)
        self.d1: float = (np.log(self.S/self.K) + (self.r - self.q + ((self.sigma**2)/2))*self.T) / (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma*np.sqrt(self.T)
        self.deltaCall = np.exp(self.q*(-1)*self.T) * norm.cdf(self.d1)
        self.deltaPut =  -np.exp(self.q*(-1)*self.T) * norm.cdf(-self.d1)
        self.gamma = (np.exp(-self.q*self.T) * norm.pdf(self.d1)) / (self.sigma * self.S * np.sqrt(self.T))
        self.vega = self.S*np.exp(self.q*(-1)*self.T)*np.sqrt(self.T)*norm.pdf(self.d1)
        self.thetaCall = None
    
    def _call(self) -> float:
         return self.S * np.exp(self.q * (-1) * self.T) * norm.cdf(self.d1) - self.K * np.exp(self.r * (-1) * self.T) * norm.cdf(self.d2)
    
    def _put(self) -> float:
         return self.K * np.exp(self.r * (-1) * self.T) * norm.cdf(self.d2 * (-1)) - self.S * np.exp(self.q * (-1) * self.T) * norm.cdf(self.d1 * (-1))
    
    def _priceNewton(self, sigma):
        d1: float = (np.log(self.S/self.K) + (self.r - self.q + ((sigma**2)/2))*self.T) / (sigma * np.sqrt(self.T))
        d2 = d1 - sigma*np.sqrt(self.T)
        if self.call:
            return self.S * np.exp(self.q * (-1) * self.T) * norm.cdf(d1) - self.K * np.exp(self.r * (-1) * self.T) * norm.cdf(d2)
        if self.put:
            return self.K * np.exp(self.r * (-1) * self.T) * norm.cdf(d2 * (-1)) - self.S * np.exp(self.q * (-1) * self.T) * norm.cdf(d1 * (-1))
    
    def _vegaNewton(self, sigma):
        d1: float = (np.log(self.S/self.K) + (self.r - self.q + ((sigma**2)/2))*self.T) / (sigma * np.sqrt(self.T))
        return self.S*np.exp(self.q*(-1)*self.T)*np.sqrt(self.T)*norm.pdf(d1)
    
    def getPrice(self) -> float:
            if self.call:
                return self.S * np.exp(self.q * (-1) * self.T) * norm.cdf(self.d1) - self.K * np.exp(self.r * (-1) * self.T) * norm.cdf(self.d2)
            if self.put:
                return self.K * np.exp(self.r * (-1) * self.T) * norm.cdf(self.d2 * (-1)) - self.S * np.exp(self.q * (-1) * self.T) * norm.cdf(self.d1 * (-1))
            
    def straddleATM(self) -> float:
        # Or use the approximation = .8 * self.sigma * np.sqrt(self.T) * self.S
        return self._call() + self._put()
    
    def comboValue(self) -> None:
        print(f"C - P = {(self._call() - self._put()).round(3)}")
        print(f"F - K = {np.round((((self.S * (1+self.r*self.T) - self.q) - self.K) / (1+self.r*self.T)),3)}")

    def impliedIR(self) -> float:
        return ((self._call() - self._put() - self.S + self.K + self.q) / self.K) / self.T
    
    def impliedDividend(self) -> float:
        return self.S - self._call() + self._put() - self.K + (self.K*self.r*self.T)
    
    def probabilityITM(self) -> float:
        return norm.cdf(self.d2)
    
    def efficiencyRatio(self) -> float:
        return abs(self.gamma / self.thetaCall)