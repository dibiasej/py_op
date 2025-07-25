from calc_engine.time_series import realized_volatility as rv
from calc_engine.volatility import iv_calc
from utils import date_utils as du
from data.option_data import OptionFactory
from data.price_data import process_price_data as p_data

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

class RealizedVolatilityFactory:
    def __init__(self):
        self.r_vol_factory = {
            "Close To Close": rv.CloseToClose(),
            "Rogers Satchell": rv.RogerSatchell(),
            "Garman Klass": rv.GarmanKlass(),
            "Yang Zhang": rv.YangZhang()
        }

    def create_realized_volatility_object(self, ticker: str, start_date: str, end_date: str, r_vol_method: str, r_vol_rolling_window, r_vol_freq) -> list[float]:
        r_vol_calc = self.r_vol_factory[r_vol_method]
        return r_vol_calc.calculate(ticker, start_date, end_date, realized_volatility_period = r_vol_rolling_window, freq = r_vol_freq)

class ImpliedVolatilityFactory:
    def __init__(self):
        self.iv_factory = {
            "Market Implied Volatility": None,
            "Newton Raphson" : iv_calc.NewtonsMethod(),
            "Bisection Method": iv_calc.BisectionMethod(),
            "Root Finder": iv_calc.RootFinder()
        }
    
    def create_implied_volatility_object(self, ticker: str, start_date: str, end_date: str, dtes: int, iv_method: str, frequency: str = "D", option_type='call') -> list[float]:

        options_date_range = du.option_close_date_range(start_date, end_date)
        close_prices, dates = p_data.get_close_prices(ticker, start_date, end_date, frequency)

        iv_list = []
        date_list = []

        for close_date, price in close_prices.items():
            
            close_date = str(close_date).split(" ")[0]
            if close_date in options_date_range:

                o_graph = OptionFactory().create_option_graph(ticker, close_date, option_type=option_type)
                exp = o_graph.get_expirations_from_int(dtes)
                atm_option = o_graph.get_atm_option(exp, close_date, price)

                if iv_method == "Market Implied Volatility":
                    iv = atm_option.get_implied_volatility()

                else:
                    atm_option_price = atm_option.get_price()
                    atm_option_strike = atm_option.get_strike()
                    atm_option_dte = atm_option.get_dte() / 252

                    iv = self.iv_factory[iv_method].calculate(atm_option_price, price, atm_option_strike, atm_option_dte, otype=option_type)
                
                if iv > 0.04:
                    iv_list.append(iv)
                    date_list.append(close_date)
                else:
                    iv_list.append(iv_list[-1])
                    date_list.append(close_date)

        return iv_list, date_list


def main():
    # User input for ticker and date range
    ticker = st.sidebar.text_input("Enter a Ticker")
    start_date = str(st.sidebar.date_input("Enter the Start Date ('YYYY-MM-DD')"))
    end_date = str(st.sidebar.date_input("Enter the End Date ('YYYY-MM-DD')"))

    r_vol_freq = str(st.sidebar.selectbox("Price Frequency", ["Daily", "Weekly", "Monthly"]))[0]
    r_vol_rolling_window = st.sidebar.number_input("Realized Volatility Rolling Window Length", min_value=1, value=21)

    # Sidebar checkboxes for selecting volatility types
    ctc_bool: bool = st.sidebar.checkbox("Close To Close", value=True)
    rs_bool: bool = st.sidebar.checkbox("Roger Satchell", value=False)
    gk_bool: bool = st.sidebar.checkbox("Garman Klass", value=False)
    yz_bool: bool = st.sidebar.checkbox("Yang Zhang", value=False)
    vix_bool: bool = st.sidebar.checkbox("VIX Overlay", value=False)
    iv_bool: bool = st.sidebar.checkbox("Implied Volatility Overlay", value=False)

    # Map checkboxes to realized volatility methods
    r_vol_methods = {
        "Close To Close": ctc_bool,
        "Rogers Satchell": rs_bool,
        "Garman Klass": gk_bool,
        "Yang Zhang": yz_bool
    }

    # Create a factory instance
    r_vol_factory = RealizedVolatilityFactory()

    # Store selected realized volatility data in a dictionary
    r_vol_dict = {}
    for method, is_selected in r_vol_methods.items():
        if is_selected:
            r_vol_dict[method] = r_vol_factory.create_realized_volatility_object(ticker, start_date, end_date, method, r_vol_rolling_window=r_vol_rolling_window, r_vol_freq=r_vol_freq)

    # Plot the selected volatilities
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, data in r_vol_dict.items():
        dates = pd.date_range(start = start_date, end = end_date, periods=len(data))
        ax.plot(dates, data, label=method, linewidth=2)

    if iv_bool:
        option_dte = int(st.sidebar.number_input("ATM Option DTE", min_value=1, value=21))
        iv_calc_method = str(st.sidebar.selectbox("IV Calculation Method", ["Market Implied Volatility", "Newton Raphson", "Bisection Method", "Root Finder"]))
        ivs, dates = ImpliedVolatilityFactory().create_implied_volatility_object(ticker, start_date, end_date, option_dte, iv_calc_method)
        dates2 = pd.date_range(start = start_date, end = end_date, periods=len(ivs))
        ax.plot(dates2, ivs, label = "IV", linewidth=2)
        
    if vix_bool:
        vix_close_prices, dates = p_data.get_close_prices("^VIX", start_date, end_date, freq = r_vol_freq) / 100
        dates = pd.date_range(start = start_date, end = end_date, periods=len(vix_close_prices))
        ax.plot(dates, vix_close_prices, label = "VIX", linewidth=2)

    # Customize the plot
    ax.set_title(f"Realized Volatility for {ticker}", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Volatility", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.7)

    # Render the plot in Streamlit
    st.pyplot(fig)

if __name__ == "__main__":
    main()
