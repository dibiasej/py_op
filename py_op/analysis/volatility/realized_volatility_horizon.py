from py_op.analysis.rolling_analytics.realized_volatility import get_realized_vol_strategy

def horizon_analysis(ticker: str, start: str, end: str, method: str = "close_to_close", freq="D", n_min: int = 10, n_max: int = 1500, step: int = 5):
    strategy = get_realized_vol_strategy(method)

    maxRvol, avgRvol, minRvol, ns = [], [], [], []

    for n in range(n_min, n_max, step):
        rvol, _ = strategy.calculate(ticker, start, end, n, freq)
        rvol = rvol[1:]  # keep or remove depending on preference

        ns.append(n)
        maxRvol.append(max(rvol))
        avgRvol.append(sum(rvol) / len(rvol))
        minRvol.append(min(rvol))

    return ns, minRvol, avgRvol, maxRvol