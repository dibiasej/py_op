from greek import Greek
from impliedVolatility import ImpliedVolatility
import numpy as np
from scipy.stats import norm
from typing import List
import pandas as pd
class Option(Greek, ImpliedVolatility):
    def __init__(self, S: float, K: float, T: float, sigma: float, r: float = .06, q: float = 0):
        super().__init__(S, K, T, sigma, r, q)

    def setPrice(self, price: float) -> None:
        self.price = price
    
    def arbitrageBoundaries(self) -> pd.DataFrame:  
        arrays: List[np.ndarray] = [
            np.array(["Lower Arbitrage Boundary", "Lower Arbitrage Boundary", "Upper Arbitrage Boundary", "Upper Arbitrage Boundary"]),
            np.array(["Call", "Put", "Call", "Put"])
        ]
        data = {
            "American" : [f"max[0,{self.S - self.K}, {round((self.S - self.K)/(1 + self.r * self.T) - self.q, 2)}]",
            f"max[0, {round(self.K - self.S, 2)}, {round(self.K/(1 + self.r * self.T) - self.S + self.q, 2)}]",
            self.S,
            self.K],
            "European" : [f"max[0, {round(((self.S - self.K)/(1 + self.r * self.T) - self.q), 2)}]",
                        f"max[0, {round(self.K/(1 + self.r * self.T) - self.S + self.q, 2)}]",
                        round(self.S - self.q, 2),
                        round(self.K/(1 + self.r * self.T), 2)]
        }
        print("S-K <= C-P <= S-Ke^-rT")
        print(f"{round(self.S - self.K, 3)} <= {round(self._call() - self._put(), 3)} <= {round(self.S - self.K * np.exp((-1)*self.r*self.T),3)}")
        return pd.DataFrame(data, index=arrays)
