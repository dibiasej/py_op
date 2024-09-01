from formulas import Formulas
import numpy as np
class ImpliedVolatility(Formulas):
    def __init__(self, S: float, K: float, T: float, sigma: float, r: float = .06, q: float = 0):
        super().__init__(S, K, T, sigma, r, q)

    def straddleIV(self) -> float:
        return 1.25* ((self._put() + self._call()) / self.S) * np.sqrt(self.T) * self.S
    
    def impliedVolatility(self) -> float:
        xnew: float = .25
        xold: float = 1 - xnew
        for i in range(100):
            if abs(xnew - xold) < .001:
                return xnew.round(3)
            else:
                xold = xnew
                xnew = xnew - ((self._priceNewton(xnew) - self.price) / self._vegaNewton(xnew))
                print(f"xnew is {xnew}")
        return False