from .positions import Position
from data.option_data import OptionFactory
import utils

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

def historical_option_prices(ticker: str, K: int, start_date: str, end_date: str, expiration: str, option_style: str = "european", option_type: str = "call") -> list[float]:

    close_dates: list[str] = utils.option_date_range(start_date, end_date)

    meta_option_price_list: list[int] = []

    for close_date in close_dates:
        option_price: float = OptionFactory().create_option(ticker, K, expiration, close_date, option_style, option_type).get_price()
        meta_option_price_list.append(option_price)

    return meta_option_price_list, close_dates

def position_back_test(ticker: str, start_date: str, end_date: str, position: Position):

    date_list: list = []
    position_price: float = 0
    print(f"position: {position}\n")

    for pos in position:

        strike = pos.get_strike()
        expiration = pos.get_expiration()
        option_type = pos.get_otype()
        exposure = pos.get_exposure()
        #print(f"pos {pos}\n")

        option_price, date = historical_option_prices(ticker, strike, start_date, end_date, expiration, option_type=option_type)
        op_price = np.array(option_price)

        if exposure == "long":
            position_price += op_price
        elif exposure == "short":
            position_price -= op_price
        else:
            raise ValueError("exposure must be long or short")

        date_list.append(date)

    return position_price, date_list

def plot_back_test(ticker: str, start_date: str, end_date: str, position: Position):
    backt = position_back_test(ticker, start_date, end_date, position)
    print(f"backt in back_test.py: {backt}")
    fig, ax = plt.subplots()
    ax.plot(backt[1][0], backt[0])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=12))
    plt.xticks(rotation=45)
    plt.title(f"{ticker} ")
    plt.show()
    #plt.figure()
    #plt.plot(backt[1][0], backt[0])
    #plt.xticks(rotation=45) 
    #plt.show()


def option_back_test_date_check(ticker: str, start_date: str, end_date: str, option_exp_date: str):

    option_factory = OptionFactory()

    close_date_range = utils.option_date_range(start_date, end_date)

    option_graph_list = []

    exp_list = []

    for date in close_date_range:
        print(date)

        option_graph = option_factory.create_option_graph(ticker, date)
        option_graph_list.append(option_graph)

    print(f"option graph list: {option_graph_list}\n")

    for i, option in enumerate(option_graph_list):
        exp = option.get_expirations()
        #print(f"exp: {exp}\n")
        print(f"exp: {exp}\n")

        if option_exp_date not in exp:
            print(f"exp: {exp}\n")
            print(f"in break \n")
            print(f"option node: {option}")
            #raise ValueError("The Expiration date you picked is not available across the back test date range")
            continue
        
        else:
            #print(f"not in break \n")
            #print(f"option node: {option.nodes}\n")
            exp_list.append(exp)

    return exp_list

def main():

    option_back_test_date_check("SPY", "2024-02-10", "2024-04-19", "2024-04-19")


    return None