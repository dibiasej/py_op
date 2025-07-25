import numpy as np

from data.data_processor.data_processor import MarketPriceProcessor, ModelPriceProcessor, VolatilityProcessor
from calc_engine.option_pricing.option_price_factory import OptionPriceFactory
from calc_engine.calibration.error_functions import normalized_root_mean_squared_error_matrix, sum_of_squares

class ErrorFunctionAdapter:

    def heston(self, S: float, strike_matrix: list[list[float]], market_prices: list[list[float]], dtes: list[str], model, **kwargs) -> float:
        
        data = ModelPriceProcessor(model).put_call_price_surface(S, strike_matrix, dtes, **kwargs)
        model_strikes = [np.array([strikes for strikes, _ in row]) for row in data]
        model_prices = [np.array([price for _, price in row]) for row in data]
        sse = normalized_root_mean_squared_error_matrix(market_prices, model_prices)
        return sse
    
    def sabr(self, S: float, strikes: list[float], market_ivs: list[float], dte: float, model, **kwargs) -> float:
        model_ivs = np.array(model.lognormal_vol(S, strikes, dte, **kwargs))
        sse = sum_of_squares(market_ivs, model_ivs)
        return sse
    
    def rbergomi(self, S: float, strikes: list[float], market_prices: list[float], dte: float, model, **kwargs) -> float:
        strikes, model_prices = zip(*ModelPriceProcessor(model).put_call_price_skew(S, strikes, dte, **kwargs))
        sse = sum_of_squares(market_prices, model_prices)
        return sse