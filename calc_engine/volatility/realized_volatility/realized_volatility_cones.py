import matplotlib.pyplot as plt

from .historical_price_volatility import HistoricalVolatilityStrategy

def realized_volatility_cone(historical_volatility_strategy: HistoricalVolatilityStrategy, ticker: str, start: str, end: str = None, freq: str = "D") -> (list[float], list[float], list[float]):

    vol_data = [historical_volatility_strategy.calculate(ticker, start, end, realized_volatility_period=i, freq=freq) for i in [20, 40, 60, 120]]

    max_vol = [max(vol) for vol in vol_data]
    avg_vol = [sum(vol) / len(vol) for vol in vol_data]
    min_vol = [min(vol) for vol in vol_data]

    return (max_vol, avg_vol, min_vol)

def plot_volatility_cone(historical_volatility_strategy: HistoricalVolatilityStrategy, ticker: str, start: str, end: str = None, freq: str = "D") -> (list[float], list[float], list[float]):
    vol_cone_data = realized_volatility_cone(historical_volatility_strategy, ticker, start, end=end, freq=freq)

    x = ["20 Days", "40 Days", "60 Days", "120 Days"]

    plt.plot(x, vol_cone_data[0], label='Max')
    plt.plot(x, vol_cone_data[1], label='Average')
    plt.plot(x, vol_cone_data[2], label='Min')

    plt.legend()
    plt.title(f'{historical_volatility_strategy} Realized Volatility Cone')
    plt.xlabel('Length of Realized Volatility Period')
    plt.ylabel('Realized Volatility')

    plt.tight_layout()

    plt.show()

