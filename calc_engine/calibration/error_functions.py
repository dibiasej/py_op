import numpy as np

# two main types of option optimization error functions are 1) based on price 2) based on IVs
""" Lets try to change this so for every objective function we pass in a numpy array of strikes and prices into sum of squres so it calculates 
the whole sum of squred errors of skew in a fact vectorized manner

Use ErrorFunctionAdapter class from data.data_processor to get these wokring with a specific model.
"""

def sum_of_squares(market_prices, model_prices):
    return np.sum((np.array(market_prices) - np.array(model_prices)) ** 2)

def sum_of_squares_matrix(market_prices, model_prices):
    return np.sum(np.array([np.sum((market_price - model_price)**2) for market_price, model_price in zip(market_prices, model_prices)]))

def normalized_root_mean_squared_error_matrix(market_prices, model_prices) -> float:
    nrmse_list = [np.sqrt(np.mean(((market_price - model_price) / (market_price + 1e-8)) ** 2)) for market_price, model_price in zip(market_prices, model_prices)]
    return np.sum(nrmse_list)

def root_mean_squared_error_matrix(market_prices, model_prices) -> float:
    rmse_list = [np.sqrt(np.mean((market_price - model_price) ** 2)) for market_price, model_price in zip(market_prices, model_prices)]
    return np.sum(rmse_list)