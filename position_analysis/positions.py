from calc_engine.option_pricing import analytical_solutions as an

class PositionNode:

    def __init__(self, spot: float | list = None, strike: int | list = None, expiration: float | list = None,
                 sigma: float | list = None, r: float = .02, q: float = 0, otype: str = "call", exposure: str = "long", stock_position: list = None) -> None:
        self._spot: float | list = spot
        self._strike: int | list = strike
        self._expiration: float | list = expiration
        self._sigma: float | list = sigma
        self._r: float | list = r
        self._q: float | list = q
        self._otype: str = otype
        self._exposure: str = exposure
        self._stock_position: list = stock_position

    def get_spot(self) -> float | list:
        return self._spot
    
    def set_spot(self, value: float | list) -> None:
        self._spot: float | list = value

    def get_strike(self) -> int | list:
        return self._strike
    
    def set_strike(self, value: int | list) -> None:
        self._strike: int | list = value

    def get_expiration(self) -> float | list:
        return self._expiration
    
    def set_expiration(self, value: float | list) -> None:
        self._expiration: float | list = value

    def get_sigma(self) -> int | list:
        return self._sigma
    
    def set_sigma(self, value: int | list) -> None:
        self._sigma: int | list = value

    def get_r(self) -> float | list:
        return self._r
    
    def set_r(self, value: float | list) -> None:
        self._r: float | list = value

    def get_q(self) -> float | list:
        return self._q
    
    def set_q(self, value: float | list) -> None:
        self._q: float | list = value

    def get_otype(self) -> float | list:
        return self._otype
    
    def set_otype(self, value: float | list) -> None:
        self._otype: float | list = value

    def get_exposure(self) -> float | list:
        return self._exposure
    
    def set_exposure(self, value: float | list) -> None:
        self._exposure: float | list = value

    def __str__(self):
        return f"PositionNode(strike={self._strike}, sigma={self._sigma}, otype={self._otype}, exposure={self._exposure})"


class Position:

    def __init__(self):
        self._positions: list = []

    def add(self, position_node):
        # note we may want to add a functionality for incorporating long stocks
        self._positions.append(position_node)

    def __setitem__(self, value, index):
        self._positions[index] = value

    def __getitem__(self, index):
        return self._positions[index]
"""    
    def __iter__(self):
        return PositionIterator(self._positions)
    
class PositionIterator:

    def __init__(self, positions):
        self._positions = positions
        self.index: int = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index < len(self._positions):
            val = self._positions[self.index]
            self.index += 1
            return val"""