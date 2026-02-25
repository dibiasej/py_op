import numpy as np

from calc_engine.option_pricing.option_price_factory import OptionPriceFactory
from calc_engine.calibration.error_functions import normalized_root_mean_squared_error_matrix, sum_of_squares

class ErrorFunctionAdapter:

    def heston_old(self, S: float, strike_matrix: list[list[float]], market_prices: list[list[float]], dtes: list[str], model, **kwargs) -> float:
        
        data = ModelPriceProcessor(model).put_call_price_surface(S, strike_matrix, dtes, **kwargs)
        model_strikes = [np.array([strikes for strikes, _ in row]) for row in data]
        model_prices = [np.array([price for _, price in row]) for row in data]
        sse = normalized_root_mean_squared_error_matrix(market_prices, model_prices)
        return sse
    
    def heston(self, S, strike_matrix, market_price_surface, dtes, model, otype = 'otm', **kwargs):
        assert len(strike_matrix) == len(market_price_surface) and len(strike_matrix) == len(dtes), "len of price and strike matrix must be the same length"

        matrix_prices = []

        for i in range(len(strike_matrix)):

            K = np.array(strike_matrix[i])
            T = dtes[i] / 365
            
            if otype.lower() == 'otm' or otype is None:
                model_call_prices = model.call(S, K, T, **kwargs)
                model_put_prices = model.put(S, K, T, **kwargs)
                model_prices = np.where(K >= S, model_call_prices, model_put_prices)
                matrix_prices.append(model_prices)

            elif otype.lower() == "call":
                model_prices = model.call(S, strike_matrix[i], dtes[i]/365, **kwargs)
                matrix_prices.append(model_prices)

            elif otype.lower() == "put":
                model_prices = model.put(S, strike_matrix[i], dtes[i]/365, **kwargs)
                matrix_prices.append(model_prices)

        return normalized_root_mean_squared_error_matrix(market_price_surface, matrix_prices)
    
    def sabr(self, S: float, strikes: list[float], market_ivs: list[float], dte: float, model, **kwargs) -> float:
        model_ivs = np.array(model.lognormal_vol(S, strikes, dte, **kwargs))
        sse = sum_of_squares(market_ivs, model_ivs)
        return sse
    
    def rbergomi(self, S: float, strikes: list[float], market_prices: list[float], dte: float, model, **kwargs) -> float:
        strikes, model_prices = zip(*ModelPriceProcessor(model).put_call_price_skew(S, strikes, dte, **kwargs))
        sse = sum_of_squares(market_prices, model_prices)
        return sse
    
    def svi(self, S, strikes, market_ivs, dte, model, r = 0.04, q=0, **kwargs):
        svi_ivs = model(S, strikes, r=r, q=q, T = dte, **kwargs)
        return sum_of_squares(market_ivs, svi_ivs)