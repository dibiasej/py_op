import numpy as np
import yfinance as yf
import datetime as dt
import multiprocessing
from abc import ABC, abstractmethod

#from global_variables import TODAY, TICKERS

TICKERS: list[str] = ["SPY", "QQQ", "TLT", "IEF", "SHY", "HYG", "LQD", "GLD", "SLV", "UCO", "^vix", "vt", 'XLE',
               	'XLF',	'XLU',	'XLI',	'GDX',	'XLK',	'XLV',	'XLY',	'XLP',	'XLB',	'XOP',	'IYR',	'XHB',
                'ITB',	'VNQ',	'GDXJ',	'IYE',	'OIH',	'XME',	'XRT',	'SMH',	'IBB', "VXX", "UVXY", "SVIX", 'KBE', 'KRE',	'XTL', 'IWM',
                'IJR', 'IWO', 'SPMD', 'SPEU', 'SPEM', 'SPSM', 'SPYG', 'SPYV', 'MDYG', 'MDYV', 'SLYG', 'SLYV', 'KIE',
                'XAR', 'XTN', 'XBI', 'XPH', 'XHS', 'XHE', 'XSW', 'XSD', 'TSLA', 'NVDA', 'AMZN', 'AAPL', 'AMD', 
                'BABA', 'PLTR', 'MSFT', 'INTC', 'BAC', 'META', 'GOOGL', "MARA", 'BA', 'PFE', 'NIO', 'ORCL', 'UBER', 'RIVN', 
                'SOFI', 'AVGO', 'COST', 'NFLX', 'MRNA', 'ADBE', 'COIN', 'ENPH', 'ZM', 'TQQQ', 'CVNA', 'PYPL', 'CRWD', 'LLY',
                'JPM', 'EEM', 'CRM', 'USB', 'XOM', 'DIS', 'HD', 'GS', 'RTX', 'JD', 'CVX', 'WFC', 'C', 'JNJ', 'KO', 'WMT',
                'AFRM', 'ROKU', 'PEP', 'U', 'TGT', 'SNOW', 'SHOP', 'CMG', 'LULU', 'LMT', 'MS', 'NOW', 'LYFT', 'NKE', 'CAT',
                'UNH', 'ASML', 'GM', 'TSM', 'KOLD', 'AXP', 'EWZ', 'SCHW', 'SBUX', 'MCD', 'V', 'FDX', 'PINS', 'GE', 'DAL',
                'ABBV', 'MA', 'F', 'VZ', 'ULTA', 'Z', 'RCL', 'SPOT', 'CVS', 'AAL', 'CCL', 'DASH', 'T', 'BLK', 'DPZ', 'EBAY',
                'TJX', '^NDX', '^SPX', '^RUT', 'TIP', 'MBB', "HON", "UPS", "DE", "CME", "ICE", "AMGN", "HUM", "GILD", "COP",
                "SLB", "HAL", "LVS", "MGM", "RIOT", "MSTR", "AIG", "PG", "LOW", "CSCO", "QCOM", "MU", "AMAT", "UUP", "UDN", "FXE",
                "FXB", "FXY", "JNK", "AGG", "UNG", "USO", "PPLT", "PALL", "CORN", "SOYB", "WEAT"]

TODAY: str = dt.datetime.now().strftime('%Y-%m-%d')

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

def insert_option_chain(conn, ticker, close_date, chain, option_type):
    cursor = conn.cursor()
    for expiration, data in chain.items():
        n_rows = len(data['Strike'])

        # create dte
        exp_date = dt.datetime.strptime(expiration, "%Y-%m-%d")
        close_dt = dt.datetime.strptime(close_date, "%Y-%m-%d")
        dte = (exp_date - close_dt).days

        for i in range(n_rows):
            cursor.execute("""
            INSERT INTO option_data (
                ticker, close_date, expiration_date, dte, option_type,
                strike, price, bid, ask, mid_price, yahoo_iv, volume, open_interest
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                close_date,
                expiration,
                dte,
                option_type,
                float(data['Strike'][i]),
                float(data['Price'][i]),
                float(data['Bid'][i]),
                float(data['Ask'][i]),
                (float(data['Bid'][i]) + float(data['Ask'][i])) / 2,
                float(data['Market Implied Volatility'][i]),
                float(data['Volume'][i]) if data['Volume'][i] is not None else None,
                float(data['Open Interest'][i]) if data['Open Interest'][i] is not None else None,
            ))

def download_data_to_db(conn, ticker: str):
    # This function downloads todays option chain data to our database
    call_chain, put_chain = create_option_chains(ticker)
    insert_option_chain(conn, ticker.upper(), TODAY, call_chain, 'call')
    insert_option_chain(conn, ticker.upper(), TODAY, put_chain, 'put')
    conn.commit()

class Download(ABC):

    @abstractmethod
    def normal_load() -> None:
        pass

    @abstractmethod
    def multiprocessing_load() -> None:
        pass

    @abstractmethod
    def asyncio_load() -> None:
        pass

class SingleDownload(Download):
    pass

class MultiDownload(Download):

    def __init__(self, conn) -> None:
        self.conn = conn

    def normal_load(self, tickers: list[str]) -> None:
        
        for ticker in tickers:
            print(ticker)

            download_data_to_db(self.conn, ticker)

    def multiprocessing_load(self, tickers: list[str]) -> None:

        args = [(self.conn, ticker) for ticker in tickers]

        with multiprocessing.Pool(10) as pool:

            pool.starmap(download_data_to_db, args)

    def asyncio_load(self) -> None:
        pass

def main():
    MultiDownload().normal_load(TICKERS)

    return None

if __name__ == "__main__":
    print(main())