import numpy as np

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