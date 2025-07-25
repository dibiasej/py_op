import numpy as np
import yfinance as yf
import pandas as pd

from data.price_data import load_price_data

def get_prices(ticker: str, start: str, end: str = None, freq: str = "D") -> list[float]:

    prices_dict = load_price_data.load_all_price_data()

    dates = np.array(prices_dict[ticker]['Date'], dtype='datetime64[D]')
    start_date = np.datetime64(start, 'D')
    end_date = np.datetime64(end, 'D') if end else np.datetime64('today', 'D')

    mask = (dates >= start_date) & (dates <= end_date)

    filtered = {
        key: np.array(values)[mask] if key != 'Date' else dates[mask].astype(str)
        for key, values in prices_dict[ticker].items()
    }

    if freq != 'D':
        df = pd.DataFrame(filtered)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.resample(freq).last().dropna(how='all')
        resampled = {
            'Date': df.index.strftime('%Y-%m-%d').to_numpy()
                    }
        for col in df.columns:
            resampled[col] = df[col].to_numpy()

        return resampled

    return filtered

def get_close_prices(ticker: str, start: str, end: str = None, freq: str = "D") -> list[float]:

    prices_dict = get_prices(ticker, start, end, freq)
    dates = prices_dict['Date']
    close_prices = prices_dict["Close"]

    return close_prices, dates

def get_log_rets(ticker: str, start: str, end: str = None, freq = "D") -> list[float]:

    close_prices, dates = get_close_prices(ticker, start, end, freq)

    return np.log(close_prices[1:] / close_prices[:-1]), dates

def get_high_low_rets(ticker: str, start: str, end: str = None, freq = "D") -> list[float]:
    prices_dict = get_prices(ticker, start, end, freq)
    high_prices = prices_dict["High"]
    low_prices = prices_dict["Low"]
    dates = prices_dict["Date"]

    return [float(np.log(high_price / low_price)) for high_price, low_price in zip(high_prices, low_prices)], dates

def get_close_open_rets(ticker: str, start: str, end: str = None, freq = "D") -> list[float]:
    prices_dict = get_prices(ticker, start, end, freq)
    close_prices = prices_dict["Close"]
    open_prices = prices_dict["Open"]
    dates = prices_dict["Date"]

    new = [np.log(close_price / open_price) for close_price, open_price in zip(close_prices, open_prices)]
    return new, dates

def get_open_close_rets(ticker: str, start: str, end: str = None, freq = "D") -> list[float]:
    prices_dict = get_prices(ticker, start, end, freq)
    close_prices = prices_dict["Close"]
    open_prices = prices_dict["Open"]
    dates = prices_dict["Date"]

    new = [np.log(open_prices[i] / close_prices[i - 1]) for i in range(1, len(close_prices))]

    return new, dates[1:]

def get_high_close_rets(ticker: str, start: str, end: str = None, freq = "D") -> list[float]:
    prices_dict = get_prices(ticker, start, end, freq)
    high_prices = prices_dict["High"]
    close_prices = prices_dict["Close"]
    dates = prices_dict["Date"]

    new = [np.log(high_price / close_price) for high_price, close_price in zip(high_prices, close_prices)]
    return new, dates

def get_high_open_rets(ticker: str, start: str, end: str = None, freq = "D") -> list[float]:
    prices_dict = get_prices(ticker, start, end, freq)
    high_prices = prices_dict["High"]
    open_prices = prices_dict["Open"]
    dates = prices_dict["Date"]

    new = [np.log(high_price / open_price) for high_price, open_price in zip(high_prices, open_prices)]

    return new, dates

def get_low_close_rets(ticker: str, start: str, end: str = None, freq = "D") -> list[float]:
    prices_dict = get_prices(ticker, start, end, freq)
    low_prices = prices_dict["Low"]
    close_prices = prices_dict["Close"]
    dates = prices_dict["Date"]

    new = [np.log(low_price / close_price) for low_price, close_price in zip(low_prices, close_prices)]

    return new, dates

def get_low_open_rets(ticker: str, start: str, end: str = None, freq = "D") -> list[float]:
    prices_dict = get_prices(ticker, start, end, freq)
    low_prices = prices_dict["Low"]
    open_prices = prices_dict["Open"]
    dates = prices_dict["Date"]

    new = [np.log(low_price / open_price) for low_price, open_price in zip(low_prices, open_prices)]

    return new, dates