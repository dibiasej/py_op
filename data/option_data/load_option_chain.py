import yfinance as yf
import numpy as np
import pickle
import os
import datetime as dt
import multiprocessing
import time
from abc import ABC, abstractmethod

TICKERS: list[str] = ["spy", "QQQ", "TLT", "IEF", "SHY", "HYG", "CMBS", "GLD", "SLV", "UCO", "^vix", "vt", 'XLE',
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
                'TJX', '^NDX', '^SPX', '^RUT']

TODAY: str = dt.datetime.now().strftime('%Y-%m-%d')

DIR: str = "/Users/dibia/OneDrive/Documents/Data/Options/options chain/"

#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#DIR = os.path.join(BASE_DIR, "data", "options chain")

def option_chain_directory(file_path, ticker) -> bool:

    """Method used for returning a directory for a specific tickers option chain, and creating the directory if it doesn't already exist.
    Note: this doesn't return the data, just a str representation of where the data is located."""

    #directory: str = file_path + ticker
    directory = os.path.join(file_path, ticker)

    if os.path.exists(directory):

        return directory

    else:

        os.makedirs(directory)

        print(f"Created directory {directory}")

        return directory

def create_option_chains(ticker: str) -> (dict, dict):
    """This function fecthes the option chain from yfinance then creates a dictionary to store the data in"""

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
    """This function calls the function to load the option chain, then it stores the dictionary returned from create_option_chain function in a pickle file."""

    print(f"Loading the option chain of {ticker}\n")
    
    data: (dict, dict) = create_option_chains(ticker)

    call_data: dict = data[0]
    put_data: dict = data[1]

    directory: str = option_chain_directory(DIR, ticker)

    call_file_path: str = directory + f"/{ticker}_call" + f"_{TODAY}"
    put_file_path: str = directory + f"/{ticker}_put" + f"_{TODAY}"

    #call_file_path: str = directory + f"/{ticker}_call" + f"_2024-12-24"
    #put_file_path: str = directory + f"/{ticker}_put" + f"_2024-12-24"

    with open(call_file_path, "wb") as option_chain_call_pickle_file:
        
        pickle.dump(call_data, option_chain_call_pickle_file)

    with open(put_file_path, "wb") as option_chain_put_pickle_file:
        
        pickle.dump(put_data, option_chain_put_pickle_file)

    print(f"{ticker}'s option chain has been loaded\n")
    print(f'-'*35)
    print('\n')

class Download(ABC):

    @staticmethod
    @abstractmethod
    def normal_load() -> None:
        pass

    @staticmethod
    @abstractmethod
    def multiprocessing_load() -> None:
        pass

    @staticmethod
    @abstractmethod
    def asyncio_load() -> None:
        pass

class SingleDownload(Download):

    @staticmethod
    def normal_load(ticker: str) -> None:
        load_data_to(ticker)

    @staticmethod
    def multiprocessing_load() -> None:
        pass

    @staticmethod
    def asyncio_load() -> None:
        pass

class MultiDownload(Download):

    @staticmethod
    def normal_load(tickers: list[str]) -> None:
        
        for ticker in tickers:

            load_data_to(ticker)

    @staticmethod
    def multiprocessing_load(tickers: list[str]) -> None:

        with multiprocessing.Pool(10) as pool:

            pool.map(load_data_to, tickers)

    @staticmethod
    def asyncio_load() -> None:
        pass

class BulkDownload(Download):

    @staticmethod
    def normal_load() -> None:
        
        for ticker in TICKERS:

            load_data_to(ticker)
    
    @staticmethod
    def multiprocessing_load() -> None:
            
        with multiprocessing.Pool(10) as pool:

            pool.map(load_data_to, TICKERS)

    @staticmethod
    def asyncio_load() -> None:
        pass


def load_data_from(ticker, date) -> (dict, dict):

    call_path: str = DIR + f"/{ticker}" + f"/{ticker}_call" + f"_{date}"
    put_path: str = DIR + f"/{ticker}" + f"/{ticker}_put" + f"_{date}"

    with open(call_path, 'rb') as call_pickle_file: 
        call_data = pickle.load(call_pickle_file)

    with open(put_path, 'rb') as put_pickle_file:
        put_data = pickle.load(put_pickle_file)

    return call_data, put_data

class Upload(ABC):

    @staticmethod
    @abstractmethod
    def upload_todays_chain(ticker: str) -> (dict, dict):
        pass
    
    @staticmethod
    @abstractmethod
    def upload_chain() -> (dict, dict):
        pass

class SingleUpload(Upload):
    """Class used to upload a single securities option chain data.
    This data is pulled from our directory we are not downloading any data so if the data has not been downloaded yet we cannot upload it."""

    @staticmethod
    def upload_todays_chain(ticker: str) -> (dict, dict):

        """Upload todays option chain data for puts and calls for a specific ticker."""

        option_chain: (dict, dict) = load_data_from(ticker, TODAY)

        return option_chain

    @staticmethod
    def upload_chain(ticker: str, date: str):

        """Upload a securities option chain data from a sepcific date not a specific expiration"""

        option_chain: (dict, dict) = load_data_from(ticker, date)

        return option_chain

def test() -> list[str]:

    """This function is used to test if the option data for a specific ticker has been downloaded and is in our directory."""

    idx: int = 0
    missing_tickers: list[str] = []

    for ticker in TICKERS:

        try:
            call, put = SingleUpload().upload_todays_chain(ticker)
            print(f"{ticker} has been uploaded\n")
            print(f"Length of call frame: {len(call)}")
            print(f"Length of put frame: {len(put)}")
            if len(call) == 0 or len(put) == 0:
                print(f"The option data for {ticker} is empty\n")
                print("-"*35)
                missing_tickers.append(ticker)

        except:

            print(f"could not upload {ticker}\n")
            missing_tickers.append(ticker)

        print(f'Index: {idx}\n')
        print(f"-"*35)
        idx += 1

    return missing_tickers

if __name__ == "__main__":
    #time.sleep(1*60*60)
    BulkDownload().multiprocessing_load()
    #for ticker in ['T', 'TJX', '^NDX', '^SPX', '^RUT']:
        #try:
        #SingleDownload().normal_load("T")
        #except:
            #continue
    t = test()
    print(t)
    #print(SingleUpload().upload_chain("WMT", "2025-04-16"))
    #print(SingleUpload().upload_chain("QQQ", "2024-09-27"))
    #print(SingleUpload().upload_chain("TSLA", "2024-09-27"))k,kk           