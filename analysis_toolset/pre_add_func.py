import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

#from price_data.process_price_data import get_close_prices

"""
Note: the first two functions are from Jeff Augens Vol book and are used to plot daily price spikes in std dev terms
"""

def price_spikes(historical_price_data):
    """
    historical_price_data param: price data from price_data module function get_close_prices() function
    """

    historical_prices = historical_price_data.to_numpy()
    close_price_index = historical_price_data.index

    ln_price = np.log(historical_prices[1:] / historical_prices[:-1])
    price_change = historical_prices[1:] - historical_prices[:-1]

    std_list = []
    one_std_chng_list = []
    current_spike_list = []
    date_list = []

    for i in range(len(ln_price)):
        
        if 20 + i == len(ln_price):
            break

        std_pr = ln_price[i:20 + i].std()
        std_list.append(std_pr)

        one_std_chng = historical_prices[20 + i] * std_pr
        one_std_chng_list.append(one_std_chng)

        current_spike = price_change[20 + i] / one_std_chng
        current_spike_list.append(current_spike)
        date_list.append(close_price_index[20 + i])

    return np.array(current_spike_list), date_list

def plot_price_spikes(close_prices):
    close_price_spikes, dates = price_spikes(close_prices)
    plt.figure()
    plt.bar(dates, close_price_spikes)
    plt.xticks(rotation=60)
    plt.show()

def price_spike_hist(close_prices):

    spikes, _ = price_spikes(close_prices)

    plt.figure()
    plt.hist(spikes, bins=40)
    plt.xlabel("Size of Price Spike (Daily Dollar Std Dev)")
    plt.ylabel("Frequency")
    plt.show()

def neg_pos_price_spike_hist(close_prices):

    spikes, _ = price_spikes(close_prices)

    pos_spikes_filter = np.where(spikes > 0)
    pos_spikes = spikes[pos_spikes_filter]

    neg_spikes_filter = np.where(spikes < 0)
    neg_spikes = spikes[neg_spikes_filter]

    # Create a figure and a set of subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    # Plot histogram on the first subplot
    axes[0].hist(pos_spikes, bins=40, color='green', alpha=0.7)
    axes[0].set_title('Histogram of Positive Spikes')
    axes[0].set_xlabel('Values')
    axes[0].set_ylabel('Frequency')

    # Plot histogram on the second subplot
    axes[1].hist(neg_spikes, bins=40, color='red', alpha=0.7)
    axes[1].set_title('Histogram of Negative Spikes')
    axes[1].set_xlabel('Values')
    axes[1].set_ylabel('Frequency')

    axes[0].set_xticklabels(axes[0].get_xticks(), rotation=45)
    axes[1].set_xticklabels(axes[1].get_xticks(), rotation=45)

    # Tight layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()