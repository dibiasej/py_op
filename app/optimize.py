import streamlit as st
import numpy as np

from data.option_data import OptionFactory
from data.data_processor.data_processor import MarketPriceProcessor, ModelPriceProcessor
from data.data_processor.adapters import ErrorFunctionAdapter
from calc_engine.calibration.error_functions import sum_of_squares, normalized_root_mean_squared_error_matrix
from calc_engine.calibration.model_optimization import OptimizerFactory
from calc_engine.option_pricing.option_price_factory import OptionPriceFactory
from calc_engine.volatility.iv_calc import ImpliedVolatility
from utils.util_funcs import get_stock_price

def error_function_processor_bridge(S, strikes, market_prices, T, model, **kwargs) -> float:
    model_strikes, model_prices = zip(*ModelPriceProcessor(model).put_call_price_skew(S, strikes, T, **kwargs))
    sse = sum_of_squares(market_prices, model_prices)
    return sse

def error_function_processor_bridge_heston(S, strikes, market_prices, T, model, **kwargs) -> float:
    data = ModelPriceProcessor(model).put_call_price_surface(S, strikes, T, **kwargs)
    model_strikes = [np.array([strikes for strikes, _ in row]) for row in data]
    model_prices = [np.array([price for _, price in row]) for row in data]
    sse = normalized_root_mean_squared_error_matrix(market_prices, model_prices)
    #print(f"sse: {sse}")
    return sse

def error_function_processor_bridge_sabr(S: float, strikes: list[float], market_ivs: list[float], T: float, model, **kwargs):
    model_ivs = np.array(model.lognormal_vol(S, strikes, T, **kwargs))
    sse = sum_of_squares(market_ivs, model_ivs)
    return sse

def get_method(model: str):

    model_method = None

    if model == "Heston":
        model_method = st.sidebar.selectbox("Heston Methods", ["Fast Fourier Transform", "Simulation"])

    elif model == "SABR":
        model_method = st.sidebar.selectbox("SABR Methods", ["Analytical", "Simulation"])

    elif model == "Rough Bergomi":
        model_method = st.sidebar.selectbox("Rough Bergomi Methods", ["Simulation"])

    else:
        st.markdown("### You must specify a model")

    return model_method

def main():

    ticker = st.text_input("Enter a Ticker")
    close_date = str(st.date_input("Enter the Close Date ('YYYY-MM-DD')"))
    S = get_stock_price(ticker, close_date)
    st.write(f"Spot price: {S}")

    market_price_processor = MarketPriceProcessor(ticker, close_date)

    exp_dates = market_price_processor.option_call_graph.get_expirations()
    exps = st.sidebar.multiselect("Expiration Dates", exp_dates)
    skew = market_price_processor.option_call_graph.get_skew(exps[0])

    dtes = np.array([market_price_processor.option_call_graph.get_dte_from_str(exp) for exp in exps]) / 252

    strike_arr = np.array(skew.strikes())
    strike_range = st.sidebar.slider("Strikes", int(strike_arr.min()), int(strike_arr.max()), value=(int(strike_arr.min()), int(strike_arr.max())))

    steps = st.sidebar.number_input("Step Size", 1)

    strikes = [strike for strike in strike_arr if strike_range[0] <= strike <= strike_range[1] and strike % steps == 0]
    #st.write(f"strikes from selected: {strikes}")

    model = str(st.sidebar.selectbox("Model", ["CEV", "Heston", "SABR", "Rough Bergomi"]))
    model_method = get_method(model)

    optimization_method = str(st.sidebar.selectbox("Method", ["SLSQP", "BFGS", "L-BFGS-B", "TNC", "Nelder-Mead", "Powell", "trust-constr"]))

    tol = float(st.sidebar.selectbox("tolerance", ["1e-10", "1e-4", "1e-3", "1e-2", "1e-1"]))

    option_pricing_model = OptionPriceFactory().create_model(model, model_method)

    if model == 'Heston':
        optimizer_factory = OptimizerFactory().create_optimizer(model, option_pricing_model, error_function_processor_bridge_heston)
    
        data = market_price_processor.put_call_price_surface(S, strikes[0], strikes[-1], exps, steps)
        market_strikes = [np.array([strikes for strikes, _ in row]) for row in data]
        market_prices = [np.array([price for _, price in row]) for row in data]

        if st.button("Notes"):
            st.write("The best methods with Heston:")
            st.write("1: Nelder-Mead - FFT - sum of normalized mean squared errors - tolerance of 1e-3")
            st.write("2. SLSQP using - FFT - sum of normalized mean quared errors - tolerance of 1e-10")
            st.write("3. trust-constr - FFT - sum of normalized mean quared errors - tolerance of 1e-10")
            st.write("tip: if we need to debug print sse from heston error bridge function.")

        if st.button("Display Data"):
            st.write(f"DTEs: {dtes.tolist()}")
            st.write(f"Expirations: {exps}")
            st.write(f"market strikes: {[[float(k) for k in arr] for arr in market_strikes]}")
            st.write(f"Market Prices: {[[float(price) for price in arr] for arr in market_prices]}")

        if st.button("Run Optimization"):
            params = optimizer_factory.optimize(S, market_strikes, market_prices, dtes, r = 0.04, q = 0, method=optimization_method, tol = tol)
            st.write(f"sigma: {params[0]}")
            st.write(f"v0: {params[1]}")
            st.write(f"theta: {params[2]}")
            st.write(f"kappa: {params[3]}")
            st.write(f"rho: {params[4]}")

    elif model == 'SABR':
        optimizer_factory = OptimizerFactory().create_optimizer(model, option_pricing_model, error_function_processor_bridge_sabr)
        market_strikes, market_prices = zip(*market_price_processor.put_call_price_skew(S, strikes[0], strikes[-1], exps[0]))
        iv_root_calc = ImpliedVolatility() 

        market_data = market_price_processor.put_call_price_surface(S, strikes[0], strikes[-1], exps, steps)
        market_strikes = [np.array([strikes for strikes, _ in row]) for row in market_data]
        market_prices = [np.array([price for _, price in row]) for row in market_data]
        market_ivs = [
            np.array([
                iv_root_calc.root_finder(
                    price, S, strike, dte, otype='call' if strike > S else 'put'
                )
                for strike, price in row
            ])
            for row, dte in zip(market_data, dtes)
        ]
        #market_ivs = [iv_root_calc.root_finder(price, S, strike, )]

        if st.button("Display Data"):

            st.write(f"market strikes: {market_strikes}")
            st.write(f"Market Prices: {market_prices}")

        if st.button("Run Optimization"):
            for i in range(len(market_ivs)):
                params = optimizer_factory.optimize(S, market_strikes[i], market_ivs[i], dtes[i], r = 0.04, q = 0, method=optimization_method, tol = tol)
                st.write(f"dte: {dtes[i] * 252}")
                st.write(f"Sigma_0: {params[0]} alpha: {params[1]} rho: {params[2]}")

    elif model == 'Rough Bergomi':
        optimizer_factory = OptimizerFactory().create_optimizer(model, option_pricing_model, ErrorFunctionAdapter().rbergomi)

        if st.button("Run Optimization"):
            for i in range(len(dtes)):
                print(f"exps: {exps[i]}")
                print(f"dtes: {dtes[i]}")
                market_strike, market_prices = zip(*market_price_processor.put_call_price_skew(S, strikes[0], strikes[-1], exps[i]))
                market_strike = np.array(market_strike)
                market_prices = np.array(market_prices)

                params = optimizer_factory.optimize(S, market_strike, market_prices, dtes[i])
                st.write(f"dte: {dtes[i] * 252}")
                st.write(f"xi (forward var): {params[0]} a: {params[1]} rho: {params[2]} eta (vol of vol): {params[3]} Hurst: {params[1] + .5}")
    else:
        optimizer_factory = OptimizerFactory().create_optimizer(model, option_pricing_model, error_function_processor_bridge)
        market_strikes, market_prices = zip(*market_price_processor.put_call_price_skew(S, strikes[0], strikes[-1], exps[0]))

        if st.button("Display Data"):

            st.write(f"market strikes: {market_strikes}")
            st.write(f"Market Prices: {market_prices}")

        params = optimizer_factory.optimize(S, market_strikes, market_prices, dtes[0], r = 0.04, q = 0, method=optimization_method, tol = tol)
        st.write(params)