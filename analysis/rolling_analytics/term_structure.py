import numpy as np

from trash.volatility_processor import TermStructureProcessor
from data.price_data import process_price_data
from utils import date_utils

def forward_rates():
    term_structure_processor = TermStructureProcessor(ticker, date, spot)
    pass

def volatility_ratios():
    pass

def rolling_term_structure(ticker, start, end):
    close_dates = date_utils.option_close_date_range(start, end)
    S_list, S_dates = process_price_data.get_close_prices(ticker, start, end)
    common_dates = [date for date in close_dates if date in S_dates]

    term_structures, new_dates = [], []

    for i in range(len(common_dates)):
        spot = S_list[i]
        date = common_dates[i]
        print(f"date: {date}")
        print(f"i: {i}")
        try:
            term_structure_processor = TermStructureProcessor(ticker, date, spot)
            print(f"in try term structure {i}")
        except:
            print(f"In except {i}, date {date}")
            continue 

        new_dates.append(date)
        term_structures.append(term_structure_processor)
    
    return new_dates, term_structures            

def rolling_data_calculator(ticker, start, end, atf = False):
    """
    Calculates the 10 day z score of the difference in 3 month iv and 1 month iv
    z score = (3_month_iv - 1_month_iv)
    parameters:
    option_exp: str of option expiration in format '1M', '3M', etc...
    """
    close_dates = date_utils.option_close_date_range(start, end)
    S_list, S_dates = process_price_data.get_close_prices(ticker, start, end)
    common_dates = [date for date in close_dates if date in S_dates]
    
    one_month_ivs = []
    three_month_ivs = []

    for i in range(len(common_dates)):
        spot = S_list[i]
        date = common_dates[i]
        print(f"date: {date}")
        print(f"i: {i}")
        try:
            term_structure_processor = TermStructureProcessor(ticker, date, spot)
            print(f"in try term structure {i}")
        except:
            print(f"In except {i}, date {date}")
            continue

        if atf:
            _, ivs = term_structure_processor.atf_put()

        else:

            _, ivs = term_structure_processor.atm_put()

        one_month_iv = ivs[0]
        three_month_iv = ivs[2]

        one_month_ivs.append(one_month_iv)
        three_month_ivs.append(three_month_iv)

    diffs = [three - one for three, one in zip(three_month_ivs, one_month_ivs)]
    diffs_array = np.array(diffs)
    print(f"len diff array")
    print(f"{len(diffs_array)}")
    means = []
    std_devs = []
    z_scores = []
    new_dates = []
    for i in range(len(diffs_array)-10):
        rolling_mean = np.mean(diffs_array[i:i+10])
        rolling_std_dev = np.std(diffs_array[i:i+10])
        z_score = (diffs_array[i+10] - rolling_mean) / rolling_std_dev
        means.append(rolling_mean)
        std_devs.append(rolling_std_dev)
        z_scores.append(z_score)
        new_dates.append(common_dates[i+10])

    return new_dates, z_scores, means, std_devs, one_month_ivs, three_month_ivs