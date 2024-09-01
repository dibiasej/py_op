import numpy as np
import yfinance as yf
import datetime as dt

def _t(data_freq: str) -> int:

    if len(data_freq) == 2:

        scale: int = int(list(data_freq)[0])
        period: str = list(data_freq)[1]

    else:
        
        period: str = data_freq
        scale: int = 1

    match period:

        case "D":
            
            return 252 - scale

        case "W":
            return 52 / scale
        
        case "M":
            return 12 / scale
        
        case "Y":
            return 1 / scale 

def _length(length: str, data_freq: str) -> int:

    if isinstance(length, int):

        return length

    if len(data_freq) == 2:

        scale: int = int(list(data_freq)[0])
        period: str = list(data_freq)[1]

    else:

        period: str = data_freq
        scale: int = 1

    match length:

        case "Day":

            return round(1 / scale)
        
        case "2 Day":

            return round(2 / scale)
        
        case "3 Day":

            return round(3 / scale)
        
        case "4 Day":

            return round(4 / scale)

        case "Week":

            return round(5 / scale) if period == "D" else round(1 / scale) if period == "W" else 0
        
        case "2 Week":

            return round(10 / scale) if period == "D" else round(2 / scale) if period == "W" else 0
        
        case "3 Week":

            return round(15 / scale) if period == "D" else round(3 / scale) if period == "W" else 0

        case "Month":

            return round(21 / scale) if period == "D" else round(4 / scale) if period == "W" else round(1 / scale) if period == "M" else 0

        case "2 Month":
            return round(42 / scale) if period == "D" else round(8 / scale) if period == "W" else round(2 / scale) if period == "M" else 0

        case "3 Month":
            return round(63 / scale) if period == "D" else round(12 / scale) if period == "W" else round(3 / scale) if period == "M" else 0
        
        case "4 Month":
            return round(84 / scale) if period == "D" else round(16 / scale) if period == "W" else round(4 / scale) if period == "M" else 0

        case "5 Month":
            return round(105 / scale) if period == "D" else round(20 / scale) if period == "W" else round(5 / scale) if period == "M" else 0
        
        case "6 Month":
            return round(126 / scale) if period == "D" else round(24 / scale) if period == "W" else round(6 / scale) if period == "M" else 0

        case "7 Month":
            return round(147 / scale) if period == "D" else round(28 / scale) if period == "W" else round(7 / scale) if period == "M" else 0
        
        case "9 Month":
            return round(189 / scale) if period == "D" else round(36 / scale) if period == "W" else round(9 / scale) if period == "M" else 0

        case "Year":
            return round(252 / scale) if period == "D" else round(52 / scale) if period == "W" else round(12 / scale) if period == "M" else 0

def _get_resampled_prices(ticker: str, start: str, end: str, freq: str):

    if freq == "D":

        return yf.Ticker(ticker).history(start = start, end = end)
    
    else:

        return yf.Ticker(ticker).history(start = start, end = end).resample(freq).last()
    
def _get_log_rets(prices) -> list:

    close_prices = prices["Close"].dropna()

    np_close_prices: np.ndarray = close_prices.to_numpy()
    log_rets = np.log(np_close_prices[1:] / np_close_prices[:-1])

    return log_rets

def _dates(prices, length, data_freq) -> list:

    length: int = _length(length, data_freq)

    close_prices = prices["Close"].dropna()

    dates = close_prices.index[length:][1:]

    return dates