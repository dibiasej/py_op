from formulas import Formulas
import matplotlib.pyplot as plt
import numpy as np
class Diagram(Formulas):
    def __init__(self, S: float, K: float, T: float, sigma: float, r: float, q: float):
        super().__init__(S, K, T, sigma, r, q)
    
    def plot(self, x, y):
        fig, ax1 = plt.subplots()
        ax1.plot(x, y)
        ax1.set_xlabel("Spot")
        ax1.set_ylabel("Payoff (Intrinsic Value)")
        return fig.show()
    def longCall(self, lower: float, upper: float):
        spot: np.ndarray = np.linspace(lower, upper, 80)
        y: np.ndarray = np.maximum(spot - self.K, 0)
        return self.plot(spot, y)
    
    def longPut(self, lower: float, upper: float):
        spot: np.ndarray = np.linspace(lower, upper, 80)
        y: np.ndarray = np.maximum(self.K - spot, 0)
        return self.plot(spot, y)
    
    def shortCall(self, lower: float, upper: float):
        spot: np.ndarray = np.linspace(lower, upper, 80)
        y: np.ndarray = -np.maximum(spot - self.K, 0)
        return self.plot(spot, y)

    def shortPut(self, lower: float, upper: float):
        spot: np.ndarray = np.linspace(lower, upper, 80)
        y: np.ndarray = -np.maximum(self.K - spot, 0)
        return self.plot(spot, y)
    
    def longStraddle(self, lower, upper):
        spot: np.ndarray = np.linspace(lower, upper, 80)
        y: np.ndarray = np.maximum(spot - self.K, 0) + np.maximum(self.K - spot, 0)
        return self.plot(spot, y)
    
    def shortStraddle(self, lower: float, upper: float):
        spot: np.ndarray = np.linspace(lower, upper, 80)
        y: np.ndarray = -np.maximum(spot - self.K, 0) - np.maximum(self.K - spot, 0)
        return self.plot(spot, y)
    
    def bullCallSpread(self, K2: float, lower: float, upper: float):
        spot: np.ndarray = np.linspace(lower, upper, 80)
        y: np.ndarray = np.maximum(spot - self.K, 0) - np.maximum(spot - K2, 0)
        return self.plot(spot, y)
    
    def bearCallSpread(self, K1: float, lower: float, upper: float):
        spot: np.ndarray = np.linspace(lower, upper, 80)
        y: np.ndarray = -np.maximum(spot - K1, 0) + np.maximum(spot - self.K, 0)
        return self.plot(spot, y)
    
    def straddleATM(self) -> float:
        return .8 * self.sigma * np.sqrt(self.T) * self.S