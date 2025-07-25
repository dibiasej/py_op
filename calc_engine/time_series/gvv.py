import numpy as np

from data.data_processor.data_processor import MarketPriceProcessor, VolatilityProcessor
#from calc_engine.volatility.gvv import GVV
from calc_engine.volatility.point_estimate_models import GVV

from data.price_data import process_price_data
from utils import date_utils

def _gvv_time_series_helper(ticker, start, end, option_exp = '1M', moneyness = .3, steps = 10, otm_options = True):
    """
    parameters:
    option_exp: str of option expiration in format '1M', '3M', etc...
    """
    close_dates = date_utils.option_close_date_range(start, end)
    S_list, S_dates = process_price_data.get_close_prices(ticker, start, end)
    common_dates = [date for date in close_dates if date in S_dates]

    atm_ivs = []
    spot_vol_covs = []
    vol_vols = []

    for i in range(len(common_dates)):
        spot = S_list[i]
        date = common_dates[i]
        market_processor = MarketPriceProcessor(ticker, date)
        exp = market_processor.option_call_graph.get_expirations_from_str(option_exp)
        dte = market_processor.option_call_graph.get_dte_from_str(exp)/252

        upper_spot = spot*(moneyness + 1)
        lower_spot = spot*(1 - moneyness)

        if otm_options == True:

            strikes, prices = zip(*market_processor.put_call_price_skew(spot, lower_spot, upper_spot, exp, steps))
            strikes, prices = np.array(strikes), np.array(prices)
            strikes, ivs = zip(*VolatilityProcessor().get_ivs(spot, prices, strikes, dte))
            strikes, ivs = np.array(strikes), np.array(ivs)

        else:
            call_data = market_processor.all_calls(lower_spot, upper_spot, exp, steps)
            put_data = market_processor.all_puts(lower_spot, upper_spot, exp, steps)
            strikes, ivs = VolatilityProcessor().equal_put_call_skew(spot, call_data, put_data, dte)

        model_gvv = GVV()
        atm_iv, spot_vol_cov, vol_vol = model_gvv.implied_parameters(spot, strikes, dte, ivs)

        atm_ivs.append(atm_iv)
        spot_vol_covs.append(spot_vol_cov)
        vol_vols.append(vol_vol)
    
    return common_dates, atm_ivs, spot_vol_covs, vol_vols

def implied_spot_vol_corr(ticker, start, end, option_exp = '1M', moneyness = .3, steps = 10, otm_options=True):
    common_dates, _, spot_vol_covs, _ = _gvv_time_series_helper(ticker, start, end, option_exp, moneyness, steps, otm_options)
    return common_dates, spot_vol_covs

def implied_vol_vol(ticker, start, end, option_exp = '1M', moneyness = .3, steps = 10, otm_options=True):
    common_dates, _, _, vol_vols = _gvv_time_series_helper(ticker, start, end, option_exp, moneyness, steps, otm_options)
    return common_dates, vol_vols

def implied_vol(ticker, start, end, option_exp = '1M', moneyness = .3, steps = 10, otm_options=True):
    common_dates, atm_ivs, _, _ = _gvv_time_series_helper(ticker, start, end, option_exp, moneyness, steps, otm_options)
    return common_dates, atm_ivs