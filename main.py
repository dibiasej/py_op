import matplotlib.pyplot as plt
import time
import numpy as np
import sqlite3
from typing import Dict, Any, List, Tuple

from data.option_data.download_option_chain import MultiDownload
from data.price_data.load_price_data import download_store_all_data
from utils.db_utils import get_connection
from global_variables import OPTION_DB_DIR, TICKERS, TODAY
from data.repositories.option_chain_repository import OptionChainRepository
#from services.snapshot_service import VolatilitySnapShotService2
from data.builders.option_chain_builder import create_chain
from data.price_data.load_price_data import load_all_price_data

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
    #build_spot_table_main()
    #conn = get_connection(OPTION_DB_DIR)


    #time.sleep(60*60)
    load_data()
    check_tickers()

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
    {% for post in paginator.posts %}
"""