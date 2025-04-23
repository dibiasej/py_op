
    spy = yf.Ticker("SPY")

    spy_prices = spy.history(start = "2024-01-01")
    return spy_prices