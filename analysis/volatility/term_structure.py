import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

from calc_engine.vol_engine.iv_calc import ImpliedVolatility
from calc_engine.calibration.put_call_parity import implied_rate
from analysis.time_series.realized_volatility import get_realized_vol_strategy
"""
module for term structures
We will get rid of MarketTermStructure and add it to data_processor.py
Maybe we add things like implied spot vol corr term structure, implied vol vol term structure
"""

    
def realized_vol(ticker: str, end: str = None, method: str = "close_to_close", freq: str = "D") -> tuple[list[float], list[str]]:

    if end is None:
        end_dt = datetime.today()
        end = end_dt.strftime("%Y-%m-%d")
    else:
        end_dt = datetime.strptime(end, "%Y-%m-%d")

    buffer_months = 3 if freq.upper().endswith("D") else 1
    start_dt = end_dt - relativedelta(years=2, months=buffer_months)
    start = start_dt.strftime("%Y-%m-%d")

    strategy = get_realized_vol_strategy(method)
    rvol_1m, _ = strategy.calculate(ticker, start, end, "M", freq)
    rvol_2m, _ = strategy.calculate(ticker, start, end, "2M", freq)
    rvol_3m, _ = strategy.calculate(ticker, start, end, "3M", freq)
    rvol_6m, _ = strategy.calculate(ticker, start, end, "6M", freq)
    rvol_9m, _ = strategy.calculate(ticker, start, end, "9M", freq)
    rvol_1y, _ = strategy.calculate(ticker, start, end, "Y", freq)
    rvol_2y, _ = strategy.calculate(ticker, start, end, "2Y", freq)

    vols = [rvol_1m[-1], rvol_2m[-1], rvol_3m[-1], rvol_6m[-1], rvol_9m[-1], rvol_1y[-1], rvol_2y[-1]]
    tenors = ["1m", "2m", "3m", "6m", "9m", "1y", "2y"]
    return vols, tenors

