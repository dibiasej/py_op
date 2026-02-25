import numpy as np

from data.price_data.process_price_data import get_close_prices
from analysis.rolling_analytics.implied_values import constant_maturity_atm_iv

"""
This module calculates all our realized metrics.
Eventually we will move realized vol into here
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


def rolling_spot_vol_stats(ticker: str, start_date: str, end_date: str, dte: int, window: int = 30, intercept: bool = False, eps: float = 1e-12):
    """
    I Tested multiple versions of spot vol rolling statistics (Benn, Junsu, Euan) and they all give me the same values. This is the best version of the code.
    In the future we might turn this to class or strategy pattern similar to RealizedVol but for now this works very well.
    """
    ivs, iv_dates = constant_maturity_atm_iv(ticker, start_date, end_date, dte)
    spot_prices, spot_dates = get_close_prices(ticker, start_date, end_date)
    spot_map = dict(zip(spot_dates, spot_prices))

    aligned_dates, aligned_spots, aligned_ivs = [], [], []
    for d, iv in zip(iv_dates, ivs):
        if d in spot_map and iv is not None and not np.isnan(iv):
            aligned_dates.append(d)
            aligned_spots.append(spot_map[d])
            aligned_ivs.append(iv)

    aligned_spots = np.asarray(aligned_spots, dtype=float)
    aligned_ivs   = np.asarray(aligned_ivs, dtype=float)

    returns = np.log(aligned_spots[1:] / aligned_spots[:-1])
    iv_changes = aligned_ivs[1:] - aligned_ivs[:-1]

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

    # window-end dates (returns start at aligned_dates[1])
    stat_dates = aligned_dates[window:]

    return betas, covs, corrs, stat_dates