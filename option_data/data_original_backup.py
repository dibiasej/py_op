import yfinance as yf
import numpy as np
import pickle
import os
import datetime as dt
import multiprocessing
import time

def option_chain_directory(file_path, ticker) -> bool:

    directory: str = file_path + ticker

    if os.path.exists(directory):

        return directory

    else:

        os.makedirs(directory)

        print(f"Created directory {directory}")

        return directory

def fetch_option_chains(ticker: str) -> (dict, dict):

    call_data: dict = {}
    put_data: dict = {}

    security = yf.Ticker(ticker)
    options_exp = security.options

    for exp in options_exp:

        option_chain = security.option_chain(exp)

        call_chain = option_chain[0]
        put_chain = option_chain[1]

        call_data[exp] = {"Strike" : np.array(call_chain["strike"]),
                          "Price": np.array(call_chain["lastPrice"]),
                          "Bid": np.array(call_chain["bid"]),
                          "Ask": np.array(call_chain["ask"]),
                          "Market Implied Volatility" : np.array(call_chain["impliedVolatility"]),
                          "Volume": np.array(call_chain["volume"]),
                          "Open Interest": np.array(call_chain["openInterest"])}
        
        put_data[exp] = {"Strike" : np.array(put_chain["strike"]),
                          "Price": np.array(put_chain["lastPrice"]),
                          "Bid": np.array(put_chain["bid"]),
                          "Ask": np.array(put_chain["ask"]),
                          "Market Implied Volatility" : np.array(put_chain["impliedVolatility"]),
                          "Volume": np.array(put_chain["volume"]),
                          "Open Interest": np.array(put_chain["openInterest"])}
        
        np.set_printoptions(suppress=True)

    return call_data, put_data

def load_data_to(ticker):

    print(f'Loading {ticker}')

    date: str = dt.datetime.now().strftime('%Y-%m-%d')
    
    data: (dict, dict) = fetch_option_chains(ticker)

    call_data: dict = data[0]
    put_data: dict = data[1]

    directory: str = option_chain_directory("/Users/dibia/OneDrive/Documents/Data/Options/options chain/", ticker)

    call_file_path: str = directory + f"/{ticker}_call" + f"_{date}"
    put_file_path: str = directory + f"/{ticker}_put" + f"_{date}"

    print(f"downloading {ticker} call data")

    with open(call_file_path, "wb") as option_chain_call_pickle_file:
        
        pickle.dump(call_data, option_chain_call_pickle_file)

    print(f"downloading {ticker} put data")
    print("-" * 30)

    with open(put_file_path, "wb") as option_chain_put_pickle_file:
        
        pickle.dump(put_data, option_chain_put_pickle_file)

def load_data_from(ticker, date) -> (dict, dict):

    call_path: str = "/Users/dibia/OneDrive/Documents/Data/Options/options chain/" + ticker + f"/{ticker}_call" + f"_{date}"
    put_path: str = "/Users/dibia/OneDrive/Documents/Data/Options/options chain/" + ticker + f"/{ticker}_put" + f"_{date}"

    with open(call_path, 'rb') as call_pickle_file:
        call_data = pickle.load(call_pickle_file)

    with open(put_path, 'rb') as put_pickle_file:
        put_data = pickle.load(put_pickle_file)

    return call_data, put_data

if __name__ == "__main__":

    tickers = ["spy", "QQQ", "TLT", "IEF", "SHY", "HYG", "CMBS", "GLD", "SLV", "UCO", "^vix", "vt", 'XLE',
               	'XLF',	'XLU',	'XLI',	'GDX',	'XLK',	'XLV',	'XLY',	'XLP',	'XLB',	'XOP',	'IYR',	'XHB',
                'ITB',	'VNQ',	'GDXJ',	'IYE',	'OIH',	'XME',	'XRT',	'SMH',	'IBB',	'KBE',	'KRE',	'XTL', 'IWM',
                'IJR', 'IWO', 'SPMD', 'SPEU', 'SPEM', 'SPSM', 'SPYG', 'SPYV', 'MDYG', 'MDYV', 'SLYG', 'SLYV', 'KIE',
                'XAR', 'XTN', 'XBI', 'XPH', 'XWEB', 'XHS', 'XHE', 'XSW', 'XSD', 'TSLA', 'NVDA', 'AMZN', 'AAPL', 'AMD', 
                'BABA', 'PLTR', 'MSFT', 'INTC', 'BAC', 'META', 'GOOGL', "MARA", 'BA', 'PFE', 'NIO', 'ORCL', 'UBER', 'RIVN', 
                'SOFI', 'AVGO', 'COST', 'NFLX', 'MRNA', 'ADBE', 'COIN', 'ENPH', 'ZM', 'TQQQ', 'CVNA', 'PYPL', 'CRWD', 'LLY',
                'JPM', 'EEM', 'CRM', 'USB', 'XOM', 'DIS', 'HD', 'GS', 'RTX', 'JD', 'CVX', 'WFC', 'X', 'C', 'JNJ', 'KO', 'WMT',
                'AFRM', 'SQ', 'ROKU', 'PEP', 'U', 'TGT', 'SNOW', 'SHOP', 'CMG', 'LULU', 'LMT', 'MS', 'NOW', 'LYFT', 'NIKE', 'CAT',
                'UNH', 'ASML', 'GM', 'TSM', 'KOLD', 'AXP', 'EWZ', 'SCHW', 'SBUX', 'MCD', 'V', 'FDX', 'PINS', 'GE', 'DAL', 'SAVE',
                'ABBV', 'MA', 'F', 'VZ', 'ULTA', 'Z', 'RCL', 'SPOT', 'CVS', 'WBA', 'AAL', 'CCL', 'DASH', 'T', 'BLK', 'DPZ', 'EBAY',
                'TJX', '^NDX', '^SPX', '^RUT']
    
    missing_tickers = []
    
                             
    start_time = time.perf_counter()

    
    with multiprocessing.Pool(10) as pool:

        pool.map(load_data_to, tickers)

    end_time = time.perf_counter()
    print(f"Time taken to load data for: {end_time - start_time} seconds")

    """
    idx = 1

    for ticker in tickers:

        print(f"we are at idx {idx}")

        try:

            load_data_to(ticker)

        except:

            print(f"we could not load {ticker}")

        idx += 1"""
    
    # below loads one ticker

    #load_data_to('DIS')

    # below is a loop that loads every ticker for a certain date
          
    start_time = time.time()

    idx = 1

    date: str = dt.datetime.now().strftime('%Y-%m-%d')

    for ticker in tickers:
        print(f"loading {ticker}")
        print(f"we are at loop {idx}")
        #try:
        call, put = load_data_from(ticker, date)
            #print(f"{ticker} has been loaded")
        #except:
            #print(f"could not load {ticker}")
            #missing_tickers.append(ticker)
        idx += 1

        #print(call)

    end_time = time.time()

    print(f"Time taken to load data for: {end_time - start_time} seconds")
    print(missing_tickers)