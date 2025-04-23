import numpy as np

from .positions import Position, PositionNode
from calc_engine.option_pricing import analytical_solutions as an

""""
Below we have functions for calcualting the BSM price wrt a variable change represented as a np.array (spot, sigma, time, rate)
"""

def spot_change_calc(spot: float, strike: int, expiration: float, sigma: float, r: float = 0, q: 
                    float = 0, otype: str = "call", exposure: str = "long"):
    
    price_change = an.bsm_call(spot, strike, expiration, sigma, r, q) if otype == "call" else an.bsm_put(spot, strike, expiration, sigma, r, q)

    return price_change

def sigma_change_calc(spot: float, strike: int, expiration: float, sigma: float, r: float = 0, q: 
                    float = 0, otype: str = "call"):
    
    print(f"sigma: {sigma}\n")
    
    price_change = an.bsm_call(spot, strike, expiration, sigma, r, q) if otype == "call" else an.bsm_put(spot, strike, expiration, sigma, r, q)
    print(f"price_change: {price_change}\n")
    print(f"____________________________\n")
    return price_change

def rate_change_calc(spot: float, strike: int, expiration: float, sigma: float, r: float = 0, q: 
                    float = 0, otype: str = "call"):
    
    price_change = an.bsm_call(spot, strike, expiration, sigma, r, q) if otype == "call" else an.bsm_put(spot, strike, expiration, sigma, r, q)

    return price_change

def time_change_calc(spot: float, strike: int, expiration: float, sigma: float, r: float = 0, q: 
                    float = 0, otype: str = "call"):
    
    price_change = an.bsm_call(spot, strike, expiration, sigma, r, q) if otype == "call" else an.bsm_put(spot, strike, expiration, sigma, r, q)

    return price_change

def premium_calc(spot: float, strike: int, expiration: float, sigma: float, r: float = 0, q: 
                    float = 0, otype: str = "call", exposure: str = "long"):
    
    premium = an.bsm_call(spot, strike, expiration, sigma, r, q) if otype == "call" else an.bsm_put(spot, strike, expiration, sigma, r, q)

    if exposure != "long" and exposure != "short":
        exposure = input("You must enter either long or short for exposure\n would you like to go [long] or [short]?\n")

    if exposure == "long": return -premium

    else:
        return premium

class ChangeCalculator:
    
    func_dict: dict = {"spot": spot_change_calc, "sigma": sigma_change_calc, "time": time_change_calc, "rate": rate_change_calc}

    def calculate_change(self, change_variable_name: str, change_variable: np.ndarray | list, spot: float, strike: int, expiration: float, sigma: float, 
                  r: float = 0, q: float = 0, otype: str = "call", exposure: str = "long"):
        
        params_str = ["spot", "sigma", "time", "r", "q"]
        params = [spot, sigma, expiration, r, q]

        if change_variable_name in params_str:
            
            var_index = params_str.index(change_variable_name)
            params[var_index] = change_variable

        else:
            raise ValueError("Incorrect change variable name parameter")
        
        if exposure != "long" and exposure != "short":
            exposure = input("You must enter either long or short for exposure\n would you like to go [long] or [short]?\n")

        prices = self.func_dict[change_variable_name](params[0], strike, params[2], params[1], params[3], params[4], otype)

        premium = premium_calc(spot, strike, expiration, sigma, r, q, otype, exposure)

        if exposure == "long": 
            print(f"in long, exposure = {exposure}")
            return prices + premium

        else:
            print(f"in short, exposure = {exposure}")
            return -prices + premium
    
    def calculate_position_node(self, position_node: PositionNode, change_variable_name: str, change_variable: np.ndarray | list):

        spot: float = position_node.get_spot()
        exposure: str = position_node.get_exposure()

        if isinstance(position_node._stock_position, (list, np.ndarray)):
            return position_node._stock_position - spot if exposure == "long" else -position_node._stock_position + spot

        strike: int = position_node.get_strike()
        expiration: float = position_node.get_expiration()
        sigma: float = position_node.get_sigma()
        r: float = position_node.get_r()
        q: float = position_node.get_q()
        otype: str = position_node.get_otype()

        position_change = self.calculate_change(change_variable_name, change_variable, spot, strike, expiration, sigma, r, q, otype, exposure)

        #premium = premium_calc(spot, strike, expiration, sigma, r, q, otype, exposure)

        position_payoff = position_change #+ premium
        
        #print(f"premium {premium}\n")

        return position_payoff
    
    def calculate_position_node_expiration(self, position_node: PositionNode, change_variable_name: str, change_variable: np.ndarray | list):

        print(position_node)

        spot: float = position_node.get_spot()
        exposure: str = position_node.get_exposure()

        if isinstance(position_node._stock_position, (list, np.ndarray)):
            return position_node._stock_position - spot if exposure == "long" else -position_node._stock_position + spot

        strike: int = position_node.get_strike()
        expiration: float = position_node.get_expiration()
        sigma: float = position_node.get_sigma()
        r: float = position_node.get_r()
        q: float = position_node.get_q()
        otype: str = position_node.get_otype()

        position_change = self.calculate_change(change_variable_name, change_variable, spot, strike, .0001, sigma, r, q, otype, exposure)

        premium = premium_calc(spot, strike, expiration, sigma, r, q, otype, exposure)

        position_payoff = position_change + premium
        
        print(f"premium {premium}\n")
        print(f"position_change[0] {position_change[0]}\n")
        print(f"position_change[-1] {position_change[-1]}\n")

        return position_payoff
    
    def calculate_position(self, positions: Position, change_variable_name: str, change_variable: np.ndarray | list):

        prices = [self.calculate_position_node(position, change_variable_name, change_variable) for position in positions]

        payoff = prices[0]

        print(f"prices[{0}]: {prices[0]}\n")

        for p in range(1, len(prices)):
            print(p)
            print(f"prices[{p}]: {prices[p]}\n")
            payoff += prices[p]


        return payoff
    
        
    def calculate_position_expiration(self, positions: Position, change_variable_name: str, change_variable: np.ndarray | list):

        prices = [self.calculate_position_node_expiration(position, change_variable_name, change_variable) for position in positions]

        payoff = prices[0]

        print(f"prices[{0}]: {prices[0]}\n")

        for p in range(1, len(prices)):
            print(p)
            print(f"prices[{p}]: {prices[p]}\n")
            payoff += prices[p]


        return payoff
