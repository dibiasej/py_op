import numpy as np

from calc_engine.calc_utils import math_utils
from calc_engine.option_pricing import analytical_solutions as an

class LocalVolatilityMethod:

    def __init__(self, model) -> None:
        self.model = model

    def set_model(self, new_model):
        self.model = new_model
        
    def calculate(self):
        pass

class DupireFiniteDifference(LocalVolatilityMethod):

    def __init__(self, model) -> None:
        super().__init__(model)

    def calculate(self, S: float, K: float | list[float], T: float, sigma: float, r: float = 0, q: float = 0, otype: str = 'call', **kwargs) -> float:
        fin_diff = math_utils.FiniteDifference(self.model, otype=otype)

        num = fin_diff.first_order_central_time(S, K, T, sigma, r, q, **kwargs)
        denom = (1/2) * K**2 * fin_diff.second_order_central_strike(S, K, T, sigma, r, q, **kwargs)

        local_vol = np.sqrt(num / denom)

        return local_vol

class DupirePriceSurface(LocalVolatilityMethod):

    def __init__(self, model) -> None:
        super().__init__(model)

    def calculate(self, strikes: list[float], dtes: list[float], price_surface: list[list[float]]) -> list[list[float]]:

        price_surf = np.array(price_surface)
        dtes = np.array(dtes) / 252

        local_vol_surf = []
        for i in range(len(price_surf) - 1):
            local_vol_list = []
            
            for j in range(1, len(strikes) - 1):
                # Calculate time derivative
                c_t = (price_surf[i + 1][j] - price_surf[i][j]) / (dtes[i + 1] - dtes[i])

                # Calculate strike derivatives
                term1 = (price_surf[i][j + 1] - price_surf[i][j]) / (strikes[j + 1] - strikes[j])
                term2 = (price_surf[i][j] - price_surf[i][j - 1]) / (strikes[j] - strikes[j - 1])
                c_wrt_k_2 = (2 / (strikes[j + 1] - strikes[j - 1])) * (term1 - term2)

                local_vol = c_t / ((1 / 2) * strikes[j]**2 * c_wrt_k_2)
                local_vol_list.append(local_vol)

            local_vol_surf.append(local_vol_list)

        local_vol_surf = np.sqrt(np.array(local_vol_surf))

        # Fill NaN values in local_vol_surf for smoother regression
        local_vol_surf = np.nan_to_num(local_vol_surf, nan=np.nanmean(local_vol_surf))
        return local_vol_surf

class LocalVolatility:

    def __init__(self, model = an.BlackScholesMertonAnalytical()) -> None:
        self.model = model

    def dupire_finite_difference(self, S: float, K: float | list[float], T: float, sigma: float, r: float = 0, q: float = 0, otype: str = 'call', **kwargs) -> float:
        return DupireFiniteDifference(self.model).calculate(S, K, T, sigma, r, q, otype, **kwargs)
    
    def dupire_price_surface(self, strikes: list[float], dtes: list[float], price_surface: list[list[float]]) -> list[list[float]]:
        return DupirePriceSurface(self.model).calculate(strikes, dtes, price_surface)