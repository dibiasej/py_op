import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

from py_op.calc_engine.vol_engine.iv_calc import ImpliedVolatility, TermStructureCalculator
from py_op.calc_engine.vol_engine.vol_funcs import variance_swap_approximation
from py_op.analysis.rolling_analytics.realized_volatility import get_realized_vol_strategy
from py_op.data.builders.option_chain_builder import create_chain
"""
module for term structures
We will get rid of MarketTermStructure and add it to data_processor.py
Maybe we add things like implied spot vol corr term structure, implied vol vol term structure
"""
def variance_swap(ticker: str, close_date: str, moneyness=.5, dtes: list[int] = None, r: float = 0):
    chain = create_chain(ticker, close_date, moneyness=moneyness)

    if dtes is None:
        dtes = chain.get_common_dtes()
    S = chain.S

    var_swaps = []

    for dte in dtes:
        put_prices, call_prices, strikes, _ = chain.get_equal_skew_prices(dte=dte, max_days_diff=20)
        var_swap = variance_swap_approximation(S, put_prices, call_prices, strikes, dte, r)
        var_swaps.append(var_swap)

    return var_swaps, dtes

def atmf_term_structure(ticker: str, close_date: str, dtes: list[int] = None, moneyness: float = .5, r = 0, q = 0, max_diff_days: int = 50):
    chain = create_chain(ticker, close_date, moneyness=moneyness)
    S = chain.S

    if dtes is None:
        dtes = [7, 15, 30, 60, 90, 120, 150, 180, 250, 360, 450, 550, 720]

    put_prices, call_prices, new_dtes, for_strikes = chain.get_equal_term_structure_atf_prices(dtes = dtes, max_diff_days=max_diff_days)
    term_struct = TermStructureCalculator()
    Ts = np.array(new_dtes)/365
    put_ivs, call_ivs, _ = term_struct.calculate_atf_term_structure(S, for_strikes, call_prices, put_prices, Ts, r = r, q = q)
    return put_ivs, call_ivs, new_dtes
    
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

