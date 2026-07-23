import matplotlib.pyplot as plt
import time
import numpy as np
import sqlite3
from typing import Dict, Any, List, Tuple

from py_op.data.option_data.download_option_chain import MultiDownload
from py_op.data.price_data.load_price_data import download_store_all_data
from py_op.utils.db_utils import get_connection
from py_op.global_variables import OPTION_DB_DIR, TICKERS, TODAY
from py_op.data.repositories.option_chain_repository import OptionChainRepository
from py_op.data.price_data.transfer_price_data_pkl_to_database import build_spot_table_main
from py_op.data.builders.option_chain_builder import create_chain
from py_op.data.price_data.load_price_data import load_all_price_data

TEMP_TICKERS = ['KBE', 'KRE',	'XTL', 'IWM',
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

def load_data():

    print("Downloading All Option Data")

    conn = get_connection(OPTION_DB_DIR)
    MultiDownload(conn).normal_load(TICKERS)

    print("Done Downloading All Option Data")
    print("\n")

    print("Downloading All Price Data")

    download_store_all_data()

    print("Done Downloading All Price Data")
    print("\n")

    return None


def check_tickers():


    for ticker in TICKERS:

        try:
            print(f"Checking Ticker {ticker}")
            chain = create_chain(ticker, TODAY, moneyness=.2, steps=5)
            skew = chain.get_otm_skew(dte = 30)

            if len(skew) == 0:
                print(f"{ticker} Skew is empty")
                print("-"*100)
            
            print(f"{ticker} Skew {skew}")
            print("\n")

        except:
            print("\n")
            print("-"*40)
            print(f"{ticker} Might have had an issue")
            print("-"*40)
            print("\n")

    return None


def main():
    # build_spot_table_main()
    # conn = get_connection(OPTION_DB_DIR)


    # time.sleep(60*60*2)
    load_data()
    check_tickers()
    download_store_all_data()

    #print(create_chain("SPY", "2026-01-16").get_common_dtes())

    #http://localhost:8000/#vol_surface

    
    return None

if __name__ == "__main__":
    print(main())


"""
Notes:
    - We will get rid of data_processor folder but keep it now so we can borrow its code
    - We will make a SkewService class that takes as an input the OptionChainRepository class
    -- SkewService will do all the data processing
    -- SkewService should return a data structure SkewSlice or a list of SkewSlices for timeseries
"""