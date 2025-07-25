import datetime as dt

from data.price_data import process_price_data

MARKET_HOLIDAYS: list[str] = ["2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19", "2024-07-04", "2024-08-02", "2024-11-28", "2024-12-25"]

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

def realized_volatility_period_length(realized_volatility_period: str, data_freq: str) -> int:

    if isinstance(realized_volatility_period, int):

        return realized_volatility_period

    if len(data_freq) == 2:

        scale: int = int(list(data_freq)[0])
        period: str = list(data_freq)[1]

    else:

        period: str = data_freq
        scale: int = 1

    length = realized_volatility_period.lower().replace(" ", "")

    if len(length) >= 3 and isinstance(length[0], int):
        length = length[:2]

    else:
        length = length[:1]

    match length:

        case "d":

            return round(1 / scale)
        
        case "2d":

            return round(2 / scale)
        
        case "3d":

            return round(3 / scale)
        
        case "4d":

            return round(4 / scale)

        case "w":

            return round(5 / scale) if period == "D" else round(1 / scale) if period == "W" else 0
        
        case "2w":

            return round(10 / scale) if period == "D" else round(2 / scale) if period == "W" else 0
        
        case "3w":

            return round(15 / scale) if period == "D" else round(3 / scale) if period == "W" else 0

        case "m":

            return round(21 / scale) if period == "D" else round(4 / scale) if period == "W" else round(1 / scale) if period == "M" else 0

        case "2m":
            return round(42 / scale) if period == "D" else round(8 / scale) if period == "W" else round(2 / scale) if period == "M" else 0

        case "3m":
            return round(63 / scale) if period == "D" else round(12 / scale) if period == "W" else round(3 / scale) if period == "M" else 0
        
        case "4m":
            return round(84 / scale) if period == "D" else round(16 / scale) if period == "W" else round(4 / scale) if period == "M" else 0

        case "5m":
            return round(105 / scale) if period == "D" else round(20 / scale) if period == "W" else round(5 / scale) if period == "M" else 0
        
        case "6m":
            return round(126 / scale) if period == "D" else round(24 / scale) if period == "W" else round(6 / scale) if period == "M" else 0

        case "7m":
            return round(147 / scale) if period == "D" else round(28 / scale) if period == "W" else round(7 / scale) if period == "M" else 0
        
        case "9m":
            return round(189 / scale) if period == "D" else round(36 / scale) if period == "W" else round(9 / scale) if period == "M" else 0

        case "y":
            return round(252 / scale) if period == "D" else round(52 / scale) if period == "W" else round(12 / scale) if period == "M" else 0

def realized_volatility_close_price_dates(ticker: str, start: str, end: str = None, realized_volatility_period: str = "M", freq: str = "D") -> list[float]:

    prices = process_price_data.get_from_cache(ticker, start, end, freq).dropna().index

    length_int = realized_volatility_period_length(realized_volatility_period, freq)

    return prices[length_int:]