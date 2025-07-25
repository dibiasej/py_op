import numpy as np
import yfinance as yf
import pandas as pd
import datetime as dt
import pickle
import os

TICKERS: list[str] = ["SPY", "QQQ", "TLT", "IEF", "SHY", "HYG", "CMBS", "GLD", "SLV", "UCO", "^vix", "vt", 'XLE',
               	'XLF',	'XLU',	'XLI',	'GDX',	'XLK',	'XLV',	'XLY',	'XLP',	'XLB',	'XOP',	'IYR',	'XHB',
                'ITB',	'VNQ',	'GDXJ',	'IYE',	'OIH',	'XME',	'XRT',	'SMH',	'IBB',	'KBE',	'KRE',	'XTL', 'IWM',
                'IJR', 'IWO', 'SPMD', 'SPEU', 'SPEM', 'SPSM', 'SPYG', 'SPYV', 'MDYG', 'MDYV', 'SLYG', 'SLYV', 'KIE',
                'XAR', 'XTN', 'XBI', 'XPH', 'XHS', 'XHE', 'XSW', 'XSD', 'TSLA', 'NVDA', 'AMZN', 'AAPL', 'AMD', 
                'BABA', 'PLTR', 'MSFT', 'INTC', 'BAC', 'META', 'GOOGL', "MARA", 'BA', 'PFE', 'NIO', 'ORCL', 'UBER', 'RIVN', 
                'SOFI', 'AVGO', 'COST', 'NFLX', 'MRNA', 'ADBE', 'COIN', 'ENPH', 'ZM', 'TQQQ', 'CVNA', 'PYPL', 'CRWD', 'LLY',
                'JPM', 'EEM', 'CRM', 'USB', 'XOM', 'DIS', 'HD', 'GS', 'RTX', 'JD', 'CVX', 'WFC', 'X', 'C', 'JNJ', 'KO', 'WMT',
                'AFRM', 'ROKU', 'PEP', 'U', 'TGT', 'SNOW', 'SHOP', 'CMG', 'LULU', 'LMT', 'MS', 'NOW', 'LYFT', 'NKE', 'CAT',
                'UNH', 'ASML', 'GM', 'TSM', 'KOLD', 'AXP', 'EWZ', 'SCHW', 'SBUX', 'MCD', 'V', 'FDX', 'PINS', 'GE', 'DAL',
                'ABBV', 'MA', 'F', 'VZ', 'ULTA', 'Z', 'RCL', 'SPOT', 'CVS', 'WBA', 'AAL', 'CCL', 'DASH', 'T', 'BLK', 'DPZ', 'EBAY',
                'TJX', '^NDX', '^SPX', '^RUT', "^VIX3M", "^VIX6M", "^VVIX", "^SKEW"]

TODAY: str = dt.datetime.now().strftime('%Y-%m-%d')

DIR: str = rf"C:\Users\dibia\OneDrive\Documents\Projects\Beta\py_op_beta\data\price_data"

def download_historical_price_data(ticker: str, freq: str = "D"):
    """
    data struct: dict{"ticker": [(date, price)]}
    """
    price_history = yf.Ticker(ticker).history(period='max')
    price_history = price_history.resample(freq).last()
    price_history.dropna(how='all', inplace = True)

    price_history.index = price_history.index.tz_localize(None)
    price_history = price_history.reset_index()
    price_history['Date'] = price_history['Date'].dt.strftime('%Y-%m-%d')
    price_history = price_history[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    price_history_dict = {
        "Date": np.array(price_history["Date"]),
        "Open": np.array(price_history['Open']),
        "High": np.array(price_history['High']),
        "Low": np.array(price_history['Low']),
        "Close": np.array(price_history['Close']),
        "Volume": np.array(price_history['Volume'])
    }
    return price_history_dict

def download_store_all_data():

    all_price_dict = {}

    for ticker in TICKERS:
        price_history = download_historical_price_data(ticker)
        all_price_dict[ticker] = price_history
        print(f"{ticker} Loaded")
        print(f"---------------")

    path = os.path.join(DIR, "all_price_data.pkl")

    with open(path, 'wb') as f:
        pickle.dump(all_price_dict, f)

def load_all_price_data() -> dict:

    path = os.path.join(DIR, "all_price_data.pkl")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Pickle file not found at: {path}")

    with open(path, 'rb') as f:
        all_price_dict = pickle.load(f)

    return all_price_dict

def main():
    download_store_all_data()
    return load_all_price_data()
    #return download_historical_price_data("SPY")

if __name__ == '__main__':
    print(main())