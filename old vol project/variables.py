import numpy as np
class Variables:
    def __init__(self, S: float, K: float, T: float, sigma: float, r: float = .06, q: float = 0):
        self.S = S
        self.K = K
        self.T = T
        self.sigma = sigma
        self.r = r
        self.q = q
        self.call = False
        self.put = False
        self.price = None

    def setCall(self) -> None:
        if self.put:
            self.put = False
        self.call = True
    def setPut(self) -> None:
        if self.call:
            self.call = False
        self.put = True
    def setPrice(self, price: float) -> None:
        self.price = price