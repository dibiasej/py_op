import numpy as np

from py_op.analysis.rolling_analytics.implied_surface import RollingVolatility


def straddle_breakevens(start_date, end_date, ticker: list = None) -> dict:
    """
    This function will return a dictionary with multiple tickers and there atm straddle breakevens across multipl expirations
    """
    # default tickers
    if tickers is None:
        tickers = ["QQQ", "GLD", "SPY", "TLT", "SLV", "UCO", "PFE", "AAPL", "PLTR"]

    atm_straddle_breakeven_data = {}

    for ticker in tickers:
        rolling_vol = RollingVolatility(ticker, start_date, end_date)
        atm_straddle_breakeven_data[ticker] = {}
        for dte in [30, 60, 90, 120, 180, 270, 365]:
            ivs, dates, spots = rolling_vol.constant_maturity_atm_iv(dte)
            ivs, spots = np.array(ivs), np.array(spots)

            price_points = (4/5) * spots * ivs * np.sqrt(dte/365)
            up_vol_points = np.log(spots / (spots + price_points))
            down_vol_points = np.log(spots / (spots - price_points))

            atm_straddle_breakeven_data[ticker][dte] = (price_points, up_vol_points, down_vol_points, dates)

    return atm_straddle_breakeven_data