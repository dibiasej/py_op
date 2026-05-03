import numpy as np

from py_op.data.price_data.process_price_data import get_close_prices, get_log_rets
from py_op.analysis.rolling_analytics.implied_surface import RollingVolatility
from py_op.analysis.rolling_analytics.realized_volatility import get_realized_vol_strategy
from py_op.utils import realized_volatility_utils as rv_utils

"""
This module calculates all our realized metrics.
Eventually we will move realized vol into here
(3/1/2026) make a realized spot fixed strike vol beta (constant maturity from dpro screen shot)
"""

def realized_hurst(sig: np.ndarray[float], largest_lag: int):
    """
    sig is log vol
    """
    x = np.arange(1, largest_lag)

    assert len(sig) > len(x), "Length of log volatility must be greater than length of lags"

    var_laf_log_vol_diff = [np.mean((sig[lag:] - sig[:-lag]) ** 2) for lag in x]
    
    coeff = np.polyfit(np.log(x), np.log(var_laf_log_vol_diff), 1)
    hurst = coeff[0]/2

    return hurst

def realized_vol_of_vol(sig: np.ndarray[float], largest_lag: int):
    """
    sig is log vol
    """
    x = np.arange(1, largest_lag)

    assert len(sig) > len(x), "Length of log volatility must be greater than length of lags"

    var_laf_log_vol_diff = [np.mean((sig[lag:] - sig[:-lag]) ** 2) for lag in x]
    
    coeff = np.polyfit(np.log(x), np.log(var_laf_log_vol_diff), 1)
    nu = np.sqrt(np.exp(coeff[1]))

    return nu

def rolling_realized_volatility(ticker: str, start: str, end: str, realized_vol_strategy: str = "close_to_close", realized_volatility_period: str = "M", freq: str = "D"):
    rvol, dates =  get_realized_vol_strategy(realized_vol_strategy).calculate(ticker, start, end, realized_volatility_period, freq)
    return rvol

def rolling_realized_skewness(ticker: str, start_date: str, end_date: str, realized_volatility_period: str = "M", freq: str = "D"):
    length_int = rv_utils.realized_volatility_period_length(realized_volatility_period, freq)
    log_rets, dates = get_log_rets(ticker, start=start_date, end=end_date, freq=freq)

    log_rets = np.asarray(log_rets, dtype=float)
    realized_skews = []

    for i in range(len(log_rets) - length_int + 1):
        x = log_rets[i:i + length_int]
        n = len(x)

        sum_sq = np.sum(x**2)

        # Avoid division by zero
        if sum_sq == 0:
            realized_skews.append(np.nan)
        else:
            sum_cu = np.sum(x**3)
            skew = (np.sqrt(n) * sum_cu) / (sum_sq ** 1.5)
            realized_skews.append(skew)

    return realized_skews, dates[length_int - 1:]

def rolling_realized_skewness1(ticker: str, start_date: str, end_date: str, realized_volatility_period: str = "M", freq: str = "D"):
    length_int = rv_utils.realized_volatility_period_length(realized_volatility_period, freq)
    log_rets, dates = get_log_rets(ticker, start=start_date, end=end_date, freq=freq)


    skews = []
    for i in range(len(log_rets) - length_int + 1):
        x = log_rets[i:i + length_int]

        n = len(x)
        x_bar = np.mean(x)
        sigma = np.sqrt(np.sum((x - x_bar) ** 2) / n)

        if sigma == 0:
            skews.append(np.nan)
        else:
            third_moment = np.sum((x - x_bar) ** 3) / n
            skews.append(third_moment / sigma**3)

    return skews[::-1], dates[length_int - 1:]

def rolling_realized_kurtosis(ticker: str, start_date: str, end_date: str, realized_volatility_period: str = "M", freq: str = "D"):

    length_int = rv_utils.realized_volatility_period_length(realized_volatility_period, freq)

    log_rets, dates = get_log_rets(ticker, start=start_date, end=end_date, freq=freq)
    log_rets = np.asarray(log_rets[::-1], dtype=float)

    kurtosis_vals = []

    for i in range(len(log_rets) - length_int + 1):
        x = log_rets[i:i + length_int]

        n = len(x)
        x_bar = np.mean(x)
        sigma = np.sqrt(np.sum((x - x_bar) ** 2) / n)

        if sigma == 0:
            kurtosis_vals.append(np.nan)
        else:
            fourth_moment = np.sum((x - x_bar) ** 4) / n
            kurtosis_vals.append(fourth_moment / (sigma ** 4))

    return kurtosis_vals[::-1], dates[length_int - 1:]

def rolling_spot_atm_iv_stats(ticker: str, start_date: str, end_date: str, dte: int, window: int = 30, intercept: bool = False, eps: float = 1e-12):
    """
    I Tested multiple versions of spot vol rolling statistics (Benn, Junsu, Euan) and they all give me the same values. This is the best version of the code.
    In the future we might turn this to class or strategy pattern similar to RealizedVol but for now this works very well.
    This realized spot floating strike vol
    """
    ivs, iv_dates, spot_prices = RollingVolatility(ticker, start_date, end_date).constant_maturity_atm_iv(dte)
    ivs, spot_prices = np.array(ivs), np.array(spot_prices)

    returns = np.log(spot_prices[1:] / spot_prices[:-1])
    iv_changes = ivs[1:] - ivs[:-1]

    betas, covs, corrs = [], [], []

    for i in range(len(returns) - window + 1):
        r = returns[i:i+window]
        v = iv_changes[i:i+window]

        if intercept:
            r = r - r.mean()
            v = v - v.mean()

        rr = np.sum(r * r)
        vv = np.sum(v * v)
        rv = np.sum(r * v)

        beta = rv / (rr + eps)
        cov  = rv / window
        corr = rv / (np.sqrt(rr * vv) + eps)

        betas.append(beta)
        covs.append(cov)
        corrs.append(corr)

    betas = np.asarray(betas)
    covs  = np.asarray(covs)*252
    corrs = np.asarray(corrs)

    return betas, covs, corrs, iv_dates[window:]

def rolling_spot_var_swap_stats(ticker: str, start_date: str, end_date: str, dte: int, window: int = 30, intercept: bool = False, r: float = 0, eps: float = 1e-12):
    ivs, iv_dates, spot_prices = RollingVolatility(ticker, start_date, end_date).constant_maturity_variance_swap(dte, r)
    ivs, spot_prices = np.array(ivs), np.array(spot_prices)

    returns = np.log(spot_prices[1:] / spot_prices[:-1])
    iv_changes = ivs[1:] - ivs[:-1]

    betas, covs, corrs = [], [], []

    for i in range(len(returns) - window + 1):
        r = returns[i:i+window]
        v = iv_changes[i:i+window]

        if intercept:
            r = r - r.mean()
            v = v - v.mean()

        rr = np.sum(r * r)
        vv = np.sum(v * v)
        rv = np.sum(r * v)

        beta = rv / (rr + eps)
        cov  = rv / window
        corr = rv / (np.sqrt(rr * vv) + eps)

        betas.append(beta)
        covs.append(cov)
        corrs.append(corr)

    betas = np.asarray(betas)
    covs  = np.asarray(covs)*252
    corrs = np.asarray(corrs)

    return betas, covs, corrs, iv_dates[window:]

def spx_vix_beta(start_date, end_date, window=21):
    """
    This is 100% correct it matches the image from jaredhstocks tweet. When we imporve on the rolling_spot_vol_stats function we can use this to test against.
    This function pulls vix and spx log ret data and calculates a rolling beta
    """
    vix_prices, vix_dates = get_close_prices("^vix", start_date, end_date)
    vix_prices = vix_prices/100
    spx_log_rets, spx_dates = get_log_rets("^SPX", start_date, end_date)
    vix_changes = np.array(vix_prices)[1:] - np.array(vix_prices)[:-1]
    vix_changes = np.asarray(vix_changes, dtype=float)

    if len(spx_log_rets) != len(vix_changes):
        raise ValueError("spot_log_rets and vol_changes must have same length")

    beta = np.full(len(spx_log_rets), np.nan)

    for i in range(window - 1, len(spx_log_rets)):
        r = spx_log_rets[i - window + 1:i + 1]
        dv = vix_changes[i - window + 1:i + 1]

        denom = np.sum(r**2)
        beta[i] = np.sum(r * dv) / denom

    return beta, spx_dates[1:]

def vix_vvix_beta(start_date, end_date, window=21):
    vix_prices, vix_dates = get_close_prices("^vix", start_date, end_date)
    vvix_prices, vvix_dates = get_close_prices("^VVIX", start_date, end_date)

    vix_prices = np.asarray(vix_prices, dtype=float) / 100
    vvix_prices = np.asarray(vvix_prices, dtype=float) / 100

    vix_changes = vix_prices[1:] - vix_prices[:-1]
    vvix_changes = vvix_prices[1:] - vvix_prices[:-1]

    if len(vix_changes) != len(vvix_changes):
        raise ValueError("vix_changes and vvix_changes must have same length")

    out = np.full(len(vix_changes), np.nan)

    for i in range(window - 1, len(vix_changes)):
        dvix = vix_changes[i - window + 1:i + 1]
        dvvix = vvix_changes[i - window + 1:i + 1]

        denom = np.sum(dvix**2)
        out[i] = np.sum(dvix * dvvix) / denom

    return out, vix_dates[1:]