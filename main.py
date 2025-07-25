import numpy as np
import matplotlib.pyplot as plt

from calc_engine.calibration.model_optimization import HestonOptimizer, rBergomiOptimizer, SABROptimizer
from calc_engine.option_pricing.option_price_factory import OptionPriceFactory
from data.data_processor.data_processor import MarketPriceProcessor, VolatilityProcessor, ModelPriceProcessor
from utils.util_funcs import get_stock_price
from data.data_processor.skew_processor import HestonSkewStrategy, SABRSkewStrategy, ConstructSkew, rBergomiSkewStrategy


def main():

    ticker = "SPY"
    close_date = "2024-08-01"

    S = get_stock_price(ticker, close_date)
    min_strike, max_strike = 450, 650

    exps = ['2024-08-30', '2024-10-31', '2024-12-31', '2025-03-31', '2025-06-30', '2026-06-18']
    dtes = []

    market_processor = MarketPriceProcessor(ticker, close_date)

    for exp in exps:
        dte = market_processor.option_call_graph.get_dte_from_str(exp)
        dtes.append(dte)

    dtes = np.array(dtes) / 252

    market_data = market_processor.put_call_price_surface(S, min_strike, max_strike, exps)
    #print(f"Expirations: {market_processor.option_call_graph.get_expirations()}")
    market_strikes = [np.array([strikes for strikes, _ in row]) for row in market_data]
    market_prices = [np.array([price for _, price in row]) for row in market_data]

    strikes_new, prices_new = zip(*MarketPriceProcessor(ticker, close_date).put_call_price_skew(S, min_strike, max_strike, exps[2]))
    strikes_ivs, ivs_new = zip(*VolatilityProcessor().otm_put_call_skew(S, prices_new, strikes_new, dtes[2]))
    strikes_new, prices_new = np.array(strikes_new), np.array(prices_new)

    skew_constructor = ConstructSkew(rBergomiSkewStrategy)
    param_names, param_values = skew_constructor.optimize(S, strikes_new, prices_new, dtes[2], method="Powell")
    param_dict = dict(zip(param_names, param_values))
    print(f"param dict: {param_dict}")
    strikes, ivs = skew_constructor.calc_ivs(S, strikes_new, None, dtes[2], **param_dict)
    print(f"ivs: {ivs}")
    plt.plot(strikes, ivs)
    plt.show()

    return None

if __name__ == "__main__":
    print(main())