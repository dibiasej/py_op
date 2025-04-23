import numpy as np
import yfinance as yf
import datetime as dt

historical_price_cache: dict = {}

def get_from_cache(ticker: str, start: str, end:str = None, freq: str = "D"):
    if (ticker.lower(), freq, start, end) in historical_price_cache:
        return historical_price_cache[(ticker.lower(), freq, start, end)]
    
    else:
        historical_price_cache[(ticker.lower(), freq, start, end)] = yf.Ticker(ticker).history(start=start, end=end).resample(freq).last()
        return historical_price_cache[(ticker.lower(), freq, start, end)]

def get_close_prices(ticker: str, start: str, end: str = None, freq: str = "D") -> list[float]:

    prices = get_from_cache(ticker, start, end, freq)["Close"].dropna()

    return prices

def get_log_rets(ticker: str, start: str, end: str = None, freq = "D") -> list[float]:

    close_prices: list[float] = get_close_prices(ticker, start, end, freq).to_numpy()

    return np.log(close_prices[1:] / close_prices[:-1])

def get_high_low_rets(ticker: str, start: str, end: str = None, freq = "D") -> list[float]:
    prices = get_from_cache(ticker, start, end, freq)[["High", "Low"]].dropna().to_numpy()

    return [np.log(price[0]/ price[1]) for price in prices]

def get_close_open_rets(ticker: str, start: str, end: str = None, freq = "D") -> list[float]:
    prices = get_from_cache(ticker, start, end, freq)[["Close", "Open"]].dropna().to_numpy()

    return [np.log(price[0]/ price[1]) for price in prices]

def get_open_close_rets(ticker: str, start: str, end: str = None, freq = "D") -> list[float]:
    open_close_prices = get_from_cache(ticker, start, end, freq)[["Open", "Close"]].dropna().to_numpy()

    return [np.log(open_close_prices[i][0] / open_close_prices[i - 1][1]) for i in range(1, len(open_close_prices))]

def get_high_close_rets(ticker: str, start: str, end: str = None, freq = "D") -> list[float]:
    prices = get_from_cache(ticker, start, end, freq)[["High", "Close"]].dropna().to_numpy()

    return [np.log(price[0]/ price[1]) for price in prices]

def get_high_open_rets(ticker: str, start: str, end: str = None, freq = "D") -> list[float]:
    prices = get_from_cache(ticker, start, end, freq)[["High", "Open"]].dropna().to_numpy()

    return [np.log(price[0]/ price[1]) for price in prices]

def get_low_close_rets(ticker: str, start: str, end: str = None, freq = "D") -> list[float]:
    prices = get_from_cache(ticker, start, end, freq)[["Low", "Close"]].dropna().to_numpy()

    return [np.log(price[0]/ price[1]) for price in prices]

def get_low_open_rets(ticker: str, start: str, end: str = None, freq = "D") -> list[float]:
    prices = get_from_cache(ticker, start, end, freq)[["Low", "Open"]].dropna().to_numpy()

    return [np.log(price[0]/ price[1]) for price in prices]

def main():

    high_low = get_low_open_rets("SPY", start = "2023-01-01", end = "2024-02-07")
    return high_low

if __name__ == "__main__":
    print(main())