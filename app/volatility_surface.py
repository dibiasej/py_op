import numpy as np
import matplotlib.pyplot
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import scipy
import streamlit as st

from calc_engine.option_pricing import analytical_solutions as an
from calc_engine.option_pricing import FFT
from calc_engine.calibration import model_optimization as mo
from calc_engine.volatility import iv_calc
from data.option_data.process_option_chain import OptionFactory, OptionGraph
#from data.data_processor.volatility_processor import bsm_iv_generator, local_volatility_generator, market_iv_generator, market_price_generator
from utils.util_funcs import get_stock_price
from calc_engine.volatility.skew import skew_fit as sk

"""
We are going to have to completely redo this
Notes: No option for OTM puts and calls in optimizer, Need to tailor it so we can define the surface parameterization (sticky delta, strike etc, moneyness)
"""

"""
def price_surface(ticker: str, close_date: str, option_type: str):
    o_graph = OptionFactory().create_option_graph(ticker, close_date, option_type)
    price_surface = []
    for """

def proccess_surface_data(o_graph_call: OptionGraph, o_graph_put: OptionGraph, strikes: list[int], exps: list[str], steps: float) -> list[list[float]]:

    #o_graph_call = OptionFactory().create_option_graph("SPY", "2024-11-25")
    #o_graph_put = OptionFactory().create_option_graph("SPY", "2024-11-25", option_type='put')

    temp_exp = list(o_graph_call.get_expirations())[2]

    atm_option_call = o_graph_call.get_atm_option(temp_exp)
    atm_strike = atm_option_call.get_strike()

    #exp_list = ['2024-12-27', '2025-01-31', '2025-02-28', '2025-04-30', '2025-06-30', '2025-12-19']
    #strike_min = 550
    #strike_max = 650
    #steps = 5
    ##strikes = np.linspace(strike_min, strike_max, strike_max - strike_min + 1)
    strikes = [strike for strike in strikes if strike % steps == 0]

    price_surf = []
    iv_surf = []

    for exp in exps:
        price_list = []
        iv_list = []
        for strike in strikes:
            if strike < atm_strike:
                option_put = o_graph_put.nodes[(exp, strike)]
                price_list.append(option_put.get_price())
                iv_list.append(option_put.get_implied_volatility())

            elif strike > atm_strike:

                option_call = o_graph_call.nodes[(exp, strike)]
                price_list.append(option_call.get_price())
                iv_list.append(option_call.get_implied_volatility())
            else:

                option_put = o_graph_put.nodes[(exp, strike)]
                option_call = o_graph_call.nodes[(exp, strike)]

                put_price = option_put.get_price()
                call_price = option_call.get_price()
                price = (call_price + put_price) / 2

                put_iv = option_put.get_implied_volatility()
                call_iv = option_call.get_implied_volatility()
                iv = (put_iv + call_iv) / 2

                price_list.append(price)
                iv_list.append(iv)

        price_surf.append(price_list)
        iv_surf.append(iv_list)

    return price_surf, iv_surf

class ModelFactory:

    def create_model(self):
        pass

def get_method(model: str):

    model_method = None

    if model == "Black Scholes Merton":
        model_method = st.sidebar.selectbox("Black Scholes Merton Methods", ["Analytical"])

    elif model == "Heston":
        model_method = st.sidebar.selectbox("Heston Methods", ["Fast Fourier Transform"])

    elif model == "SABR":
        model_method = st.sidebar.selectbox("SABR Methods", ["Analytical"])

    elif model == "Local Volatility":
        model_method = st.sidebar.selectbox("Local Volatility Methods", ["Dupire Finite Difference", "Dupire Data"])

    else:
        st.markdown("### You must specify a model")

    return model_method

class BlackScholesVolatilitySurface:

    def calculate_surface(self, ticker, close_date, strikes, expirations, option_type):

        iv_surface = []
        dte_list = []

        for iv_skew, dte, strikes_new in bsm_iv_generator(ticker, close_date, strikes, expirations, option_type='call'):
            iv_smoothed = gaussian_filter1d(iv_skew, 2)

            iv_cubic_spline = scipy.interpolate.interp1d(strikes_new, iv_smoothed, kind="cubic", fill_value="extrapolate")
            
            strikes_interpolate = np.linspace(min(strikes_new), max(strikes_new), num=100)
            iv_new = iv_cubic_spline(strikes_interpolate)

            iv_surface.append(iv_new)
            dte_list.append(dte)

        return iv_surface, strikes_interpolate, dte_list

    def plot_surface(self, ticker: str, close_date: str, strikes: list[int], expirations: list[str], option_type: str):

        iv_surface, strikes_new, dte_list = self.calculate_surface(ticker, close_date, strikes, expirations, option_type)

        vol_surf = np.array(iv_surface)
        strikes_new = np.array(strikes_new)
        dte_list = np.array(dte_list)

        X, Y = np.meshgrid(strikes_new, dte_list)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X, Y, vol_surf, cmap='viridis', edgecolor='k')

        ax.set_xlabel('Strikes')
        ax.set_ylabel('Days to Expiry (DTE)')
        ax.set_zlabel('Implied Volatility')
        fig.colorbar(surf, shrink=0.5, aspect=10)

        plt.title('Implied Volatility Surface')
        st.pyplot(fig)

class StickyStrike:

    def calculate_surface(self, ticker: str, close_date: str, strikes: list[int]):
        
        sk.StickyStrike().fit()

class LocalVolatilitySurface:

    def calculate_surface(self, ticker, close_date, strikes, expirations, option_type):

        local_vol_surface = []
        dte_list = []

        for local_vol_skew, dte, strikes_new in local_volatility_generator(ticker, close_date, strikes, expirations, option_type):
            local_vol_smoothed = gaussian_filter1d(local_vol_skew, 2)

            local_vol_cubic_spline = scipy.interpolate.interp1d(strikes_new, local_vol_smoothed, kind="cubic", fill_value="extrapolate")
            
            strikes_interpolated = np.linspace(min(strikes_new), max(strikes_new), num=100)
            local_vol_new = local_vol_cubic_spline(strikes_interpolated)

            local_vol_surface.append(local_vol_new)
            dte_list.append(dte)

        return local_vol_surface, strikes_interpolated, dte_list

    def plot_surface(self, ticker: str, close_date: str, strikes: list[int], expirations: list[str], option_type: str):

        local_vol_surface, strikes_new, dte_list = self.calculate_surface(ticker, close_date, strikes, expirations, option_type)

        vol_surf = np.array(local_vol_surface)
        strikes_new = np.array(strikes_new)
        dte_list = np.array(dte_list)

        X, Y = np.meshgrid(strikes_new, dte_list)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X, Y, vol_surf, cmap='viridis', edgecolor='k')

        ax.set_xlabel('Strikes')
        ax.set_ylabel('Days to Expiry (DTE)')
        ax.set_zlabel('Local Volatility')
        fig.colorbar(surf, shrink=0.5, aspect=10)

        plt.title('Local Volatility Surface')
        st.pyplot(fig)

class CEVVolatilitySurface:

    def optimize(self, ticker: str, close_date: str, strikes: list[int], expirations: list[str], option_type: str):

        iv_calc_obj = iv_calc.ImpliedVolatility()

        for price_list, dte, new_strikes, atm_strike, atm_iv in market_price_generator(ticker, close_date, strikes, expirations):
            parameters = mo.CEVOptimizer().optimize(atm_strike, new_strikes, price_list, dte/252)

            sigma, beta = parameters[0], parameters[1]

            cev_prices = []
            bsm_ivs = []

            for i, strike in enumerate(new_strikes):
                cev_call = an.CEVAnalytical().call(atm_strike, strike, dte/252, sigma, beta = beta)

                bs_iv = iv_calc_obj.newtons_method(cev_call, atm_strike, strike, dte/252)
                bsm_ivs.append(bs_iv)

            yield bsm_ivs, dte, new_strikes


    def calculate_surface(self, ticker: str, close_date: str, strikes: list[int], expirations: list[str], option_type: str):
        iv_surface = []
        dte_list = []
        for bsm_ivs, dte, strikes_new in self.optimize(ticker, close_date, strikes, expirations, option_type=option_type):
            iv_smoothed = gaussian_filter1d(bsm_ivs, 2)

            iv_cubic_spline = scipy.interpolate.interp1d(strikes_new, iv_smoothed, kind="cubic", fill_value="extrapolate")
            
            strikes_interpolate = np.linspace(min(strikes_new), max(strikes_new), num=100)
            iv_new = iv_cubic_spline(strikes_interpolate)

            iv_surface.append(iv_new)
            dte_list.append(dte)

        return iv_surface, strikes_interpolate, dte_list

    def plot_surface(self, ticker: str, close_date: str, strikes: list[int], expirations: list[str], option_type: str = 'call'):
        iv_surface, strikes_new, dte_list = self.calculate_surface(ticker, close_date, strikes, expirations, option_type)

        vol_surf = np.array(iv_surface)
        strikes_new = np.array(strikes_new)
        dte_list = np.array(dte_list)

        X, Y = np.meshgrid(strikes_new, dte_list)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X, Y, vol_surf, cmap='viridis', edgecolor='k')

        ax.set_xlabel('Strikes')
        ax.set_ylabel('Days to Expiry (DTE)')
        ax.set_zlabel('Implied Volatility')
        fig.colorbar(surf, shrink=0.5, aspect=10)

        plt.title('Implied Volatility Surface')
        st.pyplot(fig)

class HestonVolatilitySurface:

    def __init__(self, model_method = FFT.HestonFFT()):
        self.model_method = model_method

    def optimize(self, ticker: str, close_date: str, strikes: list[int], expirations: list[str], option_type: str):

        iv_calc_obj = iv_calc.ImpliedVolatility()

        for price_list, dte, new_strikes, atm_strike, atm_iv in market_price_generator(ticker, close_date, strikes, expirations, option_type=option_type):
            parameters = mo.HestonOptimizer().optimize(atm_strike, new_strikes, price_list, dte/252)
            sigma, v0, theta, kappa, rho = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]
            fft_prices = self.model_method.call(atm_strike, strikes, dte/252, v0=v0, theta=theta, kappa=kappa, sigma=sigma, rho=rho)
            bsm_ivs = []

            for i, strike in enumerate(strikes):
                bsm_iv = iv_calc_obj.newtons_method(fft_prices[i], atm_strike, strike, dte/252)
                bsm_ivs.append(bsm_iv)

            yield bsm_ivs, dte, new_strikes

    def calculate_surface(self, ticker, close_date, strikes, expirations, option_type):
        iv_surface = []
        dte_list = []
        for bsm_ivs, dte, strikes_new in self.optimize(ticker, close_date, strikes, expirations, option_type=option_type):
            iv_smoothed = gaussian_filter1d(bsm_ivs, 2)

            iv_cubic_spline = scipy.interpolate.interp1d(strikes_new, iv_smoothed, kind="cubic", fill_value="extrapolate")
            
            strikes_interpolate = np.linspace(min(strikes_new), max(strikes_new), num=100)
            iv_new = iv_cubic_spline(strikes_interpolate)

            iv_surface.append(iv_new)
            dte_list.append(dte)

        return iv_surface, strikes_interpolate, dte_list
    
    def plot_surface(self, ticker, close_date, strikes, expirations, option_type='call'):
        iv_surface, strikes_new, dte_list = self.calculate_surface(ticker, close_date, strikes, expirations, option_type)

        vol_surf = np.array(iv_surface)
        strikes_new = np.array(strikes_new)
        dte_list = np.array(dte_list)

        X, Y = np.meshgrid(strikes_new, dte_list)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X, Y, vol_surf, cmap='viridis', edgecolor='k')

        ax.set_xlabel('Strikes')
        ax.set_ylabel('Days to Expiry (DTE)')
        ax.set_zlabel('Implied Volatility')
        fig.colorbar(surf, shrink=0.5, aspect=10)

        plt.title('Implied Volatility Surface')
        st.pyplot(fig)

class SABRVolatilitySurface:

    def optimize(self, ticker: str, close_date: str, strikes: list[int], expirations: list[str], model=an.SABRAnalytical(), option_type: str ='call'):
        iv_calc_obj = iv_calc.ImpliedVolatility()

        for iv_skew, dte, strikes_new, S in market_iv_generator(ticker, close_date, strikes, expirations):
            parameters = mo.SABROptimizer().optimize(S, strikes_new, iv_skew, dte/252)
            sigma_0, alpha, rho = parameters[0], parameters[1], parameters[2]
            sabr_prices = []
            bsm_ivs = []

            for strike in strikes_new:
                sabr_price = an.SABRAnalytical().call(S, strike, dte/252, sigma_0=sigma_0, r=0, q=0, alpha=alpha, beta=.99, rho=rho)
                sabr_prices.append(sabr_price)
                bsm_iv = iv_calc_obj.newtons_method(sabr_price, S, strike, dte, 0)
                bsm_ivs.append(bsm_iv)

            yield bsm_ivs, dte, strikes_new

    def calculate_surface(self, ticker, close_date, strikes, expirations, model=an.SABRAnalytical(), option_type='call'):
        iv_surface = []
        dte_list = []
        for bsm_ivs, dte, strikes_new in self.optimize(ticker, close_date, strikes, expirations, option_type=option_type):
            iv_smoothed = gaussian_filter1d(bsm_ivs, 2)

            iv_cubic_spline = scipy.interpolate.interp1d(strikes_new, iv_smoothed, kind="cubic", fill_value="extrapolate")
            
            strikes_interpolate = np.linspace(min(strikes_new), max(strikes_new), num=100)
            iv_new = iv_cubic_spline(strikes_interpolate)

            iv_surface.append(iv_new)
            dte_list.append(dte)

        return iv_surface, strikes_interpolate, dte_list

    def plot_surface(self, ticker, close_date, strikes, expirations, option_type = 'call'):
        iv_surface, strikes_new, dte_list = self.calculate_surface(ticker, close_date, strikes, expirations, option_type)

        vol_surf = np.array(iv_surface)
        strikes_new = np.array(strikes_new)
        dte_list = np.array(dte_list)

        X, Y = np.meshgrid(strikes_new, dte_list)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X, Y, vol_surf, cmap='viridis', edgecolor='k')

        ax.set_xlabel('Strikes')
        ax.set_ylabel('Days to Expiry (DTE)')
        ax.set_zlabel('Implied Volatility')
        fig.colorbar(surf, shrink=0.5, aspect=10)

        plt.title('Implied Volatility Surface')
        st.pyplot(fig)

class SurfaceFactory:

    @staticmethod
    def surface_creator(model: str):
        match model.lower():

            case "bsm" | "black scholes merton" | "black scholes":
                return BlackScholesVolatilitySurface()
            
            case "local vol" | "dupire" | "local volatility":
                return LocalVolatilitySurface()
            
            case "heston":
                return HestonVolatilitySurface()
            
            case "sabr":
                return SABRVolatilitySurface()
            
            case "cev":
                return CEVVolatilitySurface()

def main():

    ticker = st.text_input("Enter a Ticker")
    close_date = str(st.date_input("Enter the Close Date ('YYYY-MM-DD')"))

    o_graph_call = OptionFactory().create_option_graph(ticker, close_date)
    o_graph_put = OptionFactory().create_option_graph(ticker, close_date, option_type='put')

    exp_dates = o_graph_call.get_expirations()

    exps = st.sidebar.multiselect("Expiration Dates", exp_dates)
    skew = o_graph_call.get_skew(exps[0])

    strike_arr = np.array(skew.strikes())
    strike_range = st.sidebar.slider("Strikes", int(strike_arr.min()), int(strike_arr.max()), value=(int(strike_arr.min()), int(strike_arr.max())))

    steps = st.sidebar.number_input("Step Size", 1)

    strikes = [strike for strike in strike_arr if strike_range[0] <= strike <= strike_range[1] and strike % steps == 0]
    st.markdown(f"### strikes {strikes}")

    model = str(st.sidebar.selectbox("Model", ["Black Scholes Merton", "CEV", "Heston", "SABR", "Local Volatility"]))
    model_method = get_method(model)

    if ticker and close_date:
        vol_surf_obj = SurfaceFactory().surface_creator(model)

        vol_surf_obj.plot_surface(ticker, close_date, strikes, exps, option_type='call')