import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from typing import List
from variables import Variables
from formulas import Formulas
class Greek(Formulas):
    # use hash table
    def __init__(self, S: float, K: float, T: float, sigma: float, r: float, q: float):
        super().__init__(S, K, T, sigma, r, q)

    def getDelta(self) -> float:
        if self.call:
            return self.deltaCall
        if self.put:
            return self.deltaPut
    def getGamma(self) -> float:
        return self.gamma
    def getVega(self) -> float:
        return self.vega