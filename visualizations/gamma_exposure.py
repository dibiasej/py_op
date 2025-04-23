import streamlit as st
import plotly.graph_objects as go
#import matplotlib.pyplot as plt

from data.option_data import OptionFactory
from calc_engine.greeks import analytical_greeks as an

def main_button_gamma():
    #options = ['Volatility Surface', 'Back Testing', 'Toolset', 'Option 4']

    #selected_option = st.selectbox('Pick an option:', options, index=1)
    if "selected_option" not in st.session_state:
        st.session_state.selected_option = None

    # Define button actions
    if st.button("Daily Gamma Exposure By Strike"):
        st.session_state.selected_option = "Daily Gamma Exposure By Strike"
    
    if st.button("Total Rolling Gamma Exposure"):
        st.session_state.selected_option = "Total Rolling Gamma Exposure"

    # Display the current selected option
    st.write(f"Selected Option: {st.session_state.selected_option}")

    return st.session_state.selected_option

def get_gamma_exposure(ticker: str, close_date: str, expirations: str):

    gamma_calc = an.AnalyticalGamma()

    o_graph_call = OptionFactory().create_option_graph(ticker, close_date, option_type='call')
    o_graph_put = OptionFactory().create_option_graph(ticker, close_date, option_type='put')

    if "total" in expirations:
        expirations = list(o_graph_call.get_expirations())

    atm_option = o_graph_call.get_atm_option(expirations[0], close_date)
    atm_strike = atm_option.get_strike()

    call_gamma_exp_dict = {}
    put_gamma_exp_dict = {}
    strike_set = set()
    
    st.markdown(expirations)

    for exp in expirations:
        call_skew = o_graph_call.get_skew(exp)
        put_skew = o_graph_put.get_skew(exp)

        for call_option, put_option in zip(call_skew, put_skew):
            if call_option.get_strike() != put_option.get_strike():
                pass

            strike = call_option.get_strike()
            call_dte = call_option.get_dte()

            call_oi = call_option.get_open_interest()
            call_iv = call_option.get_implied_volatility()
            call_gamma = gamma_calc.calculate(atm_strike, strike, call_dte, call_iv, otype='call')

            put_oi = put_option.get_open_interest()
            put_iv = put_option.get_implied_volatility()
            put_gamma = gamma_calc.calculate(atm_strike, strike, call_dte, put_iv, otype='put')

            call_gamma_exp = call_gamma * call_oi * atm_strike * 100
            put_gamma_exp = -(put_gamma * put_oi * atm_strike * 100)

            if strike in call_gamma_exp_dict:
                call_gamma_exp_dict[strike] = call_gamma_exp_dict[strike] + call_gamma_exp
            else:
                call_gamma_exp_dict[strike] = call_gamma_exp
            
            if strike in put_gamma_exp_dict:
                put_gamma_exp_dict[strike] = put_gamma_exp_dict[strike] + put_gamma_exp
            else:
                put_gamma_exp_dict[strike] = put_gamma_exp

            strike_set.add(strike)

    return call_gamma_exp_dict, put_gamma_exp_dict, sorted(strike_set)

def daily_gamma_exposure_strike(ticker: str, exp: str):#, o_graph_call, o_graph_put):

    close_date = str(st.date_input("Enter the Close Date ('YYYY-MM-DD')"))

    call_gamma_dict, put_gamma_dict, strikes = get_gamma_exposure(ticker, close_date, exp)
    strikes = list(strikes)

    exp_dates = list(o_graph_call.get_expirations())

    exp_dates.append("total")

    exps = st.sidebar.multiselect("Expiration Dates", exp_dates)

    call_gamma = list(call_gamma_dict.values())
    put_gamma = list(put_gamma_dict.values())

    show_calls = st.sidebar.checkbox("Show Call Gamma", value=True)
    show_puts = st.sidebar.checkbox("Show Put Gamma", value=True)
    show_total = st.sidebar.checkbox("Show Total Gamma", value=False)

    fig = go.Figure()

    if show_calls:
        fig.add_trace(go.Bar(x=call_gamma, y=strikes, name="Call Gamma", marker_color="blue", orientation='h'))

    if show_puts:
        fig.add_trace(go.Bar(x=put_gamma, y=strikes, name="Put Gamma", marker_color="red", orientation='h'))

    if show_total:
        total_gamma = [c + p for c, p in zip(call_gamma, put_gamma)]
        fig.add_trace(go.Bar(x=total_gamma, y=strikes, name="Total Gamma", marker_color="green", orientation='h'))

    fig.update_layout(
        title=f"Gamma Exposure for {ticker} (Expiration: {exp})",
        xaxis_title="Gamma Exposure",
        yaxis_title="Strikes",
        barmode="group",  # Group bars side by side
        template="plotly_white",
        legend_title="Gamma Types"
    )

    st.plotly_chart(fig, use_container_width=True)

SCREEN_FACTORY_GAMMA = {"Daily Gamma Exposure By Strike" : daily_gamma_exposure_strike
                        }

def main():

    st.session_state['current_gamma_screen'] = main_button_gamma()

    display_object = SCREEN_FACTORY_GAMMA[str(st.session_state["current_gamma_screen"])]

    ticker = st.text_input("Enter a Ticker")
    #close_date = str(st.date_input("Enter the Close Date ('YYYY-MM-DD')"))

    #o_graph_call = OptionFactory().create_option_graph(ticker, close_date, option_type='call')
    #o_graph_put = OptionFactory().create_option_graph(ticker, close_date, option_type='put')

    exp_dates = list(o_graph_call.get_expirations())

    exp_dates.append("total")

    exps = st.sidebar.multiselect("Expiration Dates", exp_dates)

    st.markdown(exps)

    if ticker and close_date:
        plot_gamma_exposure(ticker, close_date, exps, o_graph_call, o_graph_put)

    pass