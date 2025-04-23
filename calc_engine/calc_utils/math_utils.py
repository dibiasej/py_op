
"""
As of right now we have our Finitie Difference method in here but we might change it to somewhere else later.

"""

class FiniteDifference:

    def __init__(self, model, otype: str = 'call') -> None:
        self.model = model
        self.otype: str = otype

    def first_order_central_time(self, S: float, K: int, T: float, sigma: float, r: float = 0, q: float = 0, **kwargs) -> None:
        h = 1/252

        if self.otype == 'call':
            upper = self.model.call(S, K, T+h, sigma, r, q, **kwargs) 
            lower = self.model.call(S, K, T-h, sigma, r, q, **kwargs)

        else:
            upper = self.model.put(S, K, T+h, sigma, r, q, **kwargs) 
            lower = self.model.put(S, K, T-h, sigma, r, q, **kwargs)

        return (upper - lower) / (2*h)
    
    def second_order_central_strike(self, S: float, K: int, T: float, sigma: float, r: float = 0, q: float = 0, **kwargs) -> float | list[float]:
        h = 1

        if self.otype == 'call':
            upper = self.model.call(S, K+h, T, sigma, r, q, **kwargs)
            lower = self.model.call(S, K-h, T, sigma, r, q, **kwargs)
            middle = self.model.call(S, K, T, sigma, r, q, **kwargs)

        else:
            upper = self.model.put(S, K+h, T, sigma, r, q, **kwargs)
            lower = self.model.put(S, K-h, T, sigma, r, q, **kwargs)
            middle = self.model.put(S, K, T, sigma, r, q, **kwargs)

        return (lower - 2*middle + upper) / h**2