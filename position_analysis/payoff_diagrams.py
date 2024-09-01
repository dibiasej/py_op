import numpy as np
import matplotlib.pyplot as plt

from .positions import Position, PositionNode
from .position_calc import ChangeCalculator
from calc_engine.option_pricing import analytical_solutions as an

class PayoffDiagram:

    def __init__(self, payoff_diagram) -> None:
        self.payoff_diagram = payoff_diagram

    def plot(self, title: bool = False, xlabel: bool = False, ylabel: bool = False):

        y = self.payoff_diagram.get_payoff()
        x = self.payoff_diagram.get_change_variable()

        plt.figure()
        plt.plot(x, y)
        plt.show()

    def plot_exp(self, title: bool = False, xlabel: bool = False, ylabel: bool = False):

        y = self.payoff_diagram.get_payoff()
        y_exp = self.payoff_diagram.get_payoff_exp()
        x = self.payoff_diagram.get_change_variable()

        print(f"y expiration: {y_exp}\n")
        print(f"y: {y}\n")

        plt.figure()
        plt.plot(x, y, 'r')
        plt.plot(x, y_exp, 'b')
        plt.show()

    def plot_expiration(self):
        pass

class PayoffSpot:

    def __init__(self) -> None:
        self.payoff: list | np.ndarray = None 
        self.change_variable: list | np.ndarray = None
        self.payoff_exp: list | np.ndarray = None

    def get_payoff(self):
        return self.payoff
    
    def get_change_variable(self):
        return self.change_variable
    
    def get_payoff_exp(self):
        return self.payoff_exp

    def set_payoff(self, change_variable: np.ndarray | list, spot: float, strike: int, expiration: float, sigma: float, r: float = 0, q: float = 0, otype: str = "call", exposure: str = "long"):

        calculator = ChangeCalculator()

        self.payoff = calculator.calculate_change("spot", change_variable, spot, strike, expiration, sigma, r, q, otype, exposure)

        self.change_variable = change_variable

        #payoff_diagram = PayoffDiagram().plot(change_variable, payoff)

    def set_position_payoff(self, position: Position, change_variable: list | np.ndarray):
        
        calculator = ChangeCalculator()

        position_payoff = calculator.calculate_position(position, "spot", change_variable)

        position_payoff_exp = calculator.calculate_position_expiration(position, "spot", change_variable)

        self.payoff = position_payoff

        self.change_variable = change_variable

        self.payoff_exp = position_payoff_exp

        #payoff_diagram = PayoffDiagram().plot(change_variable, position_payoff)

class PayoffSigma:

    def __init__(self) -> None:
        self.payoff: list | np.ndarray = None 
        self.change_variable: list | np.ndarray = None
        self.payoff_exp: list | np.ndarray = None

    def get_payoff(self):
        return self.payoff
    
    def get_change_variable(self):
        return self.change_variable
    
    def get_payoff_exp(self):
        return self.payoff_exp
    
    def set_payoff(self, change_variable: np.ndarray | list, spot: float, strike: int, expiration: float, sigma: float, r: float = 0, q: float = 0, otype: str = "call", exposure: str = "long"):

        calculator = ChangeCalculator()

        self.payoff = calculator.calculate_change("sigma", change_variable, spot, strike, expiration, sigma, r, q, otype, exposure)

        self.change_variable = change_variable

    def set_position_payoff(self, position: Position, change_variable: list | np.ndarray):
        
        calculator = ChangeCalculator()

        position_payoff = calculator.calculate_position(position, "sigma", change_variable)

        position_payoff_exp = calculator.calculate_position_expiration(position, "sigma", change_variable)

        self.payoff = position_payoff

        self.change_variable = change_variable

        self.payoff_exp = position_payoff_exp

class PayoffTime:

    def __init__(self) -> None:
        self.payoff: list | np.ndarray = None 
        self.change_variable: list | np.ndarray = None
        self.payoff_exp: list | np.ndarray = None

    def get_payoff(self):
        return self.payoff
    
    def get_change_variable(self):
        return self.change_variable
    
    def get_payoff_exp(self):
        return self.payoff_exp
    
    def set_payoff(self, change_variable: np.ndarray | list, spot: float, strike: int, expiration: float, sigma: float, r: float = 0, q: float = 0, otype: str = "call", exposure: str = "long"):

        calculator = ChangeCalculator()

        self.payoff = calculator.calculate_change("time", change_variable, spot, strike, expiration, sigma, r, q, otype, exposure)

        self.change_variable = change_variable

    def set_position_payoff(self, position: Position, change_variable: list | np.ndarray):
        
        calculator = ChangeCalculator()

        position_payoff = calculator.calculate_position(position, "time", change_variable)

        position_payoff_exp = calculator.calculate_position_expiration(position, "time", change_variable)

        self.payoff = position_payoff

        self.change_variable = change_variable

        self.payoff_exp = position_payoff_exp