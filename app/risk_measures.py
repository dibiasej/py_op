import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from position_analysis.positions import Portfolio
from calc_engine.option_pricing.option_price_factory import OptionPriceFactory 

def add_position():
    """Adds a new position to the session state."""
    pos_id = len(st.session_state["positions"]) + 1
    st.session_state["positions"].append(pos_id)
    st.session_state["selected_position"][pos_id] = None  # No type selected yet

def select_stock(pos_id):
    """Marks a position as a Stock."""
    st.session_state["selected_position"][pos_id] = "Stock"

def select_option(pos_id):
    """Marks a position as an Option."""
    st.session_state["selected_position"][pos_id] = "Option"

def display_positions(portfolio):
    """Displays all positions in the session state."""
    for i, pos_id in enumerate(st.session_state["positions"]):
        with st.container():
            st.write(f"### Position {pos_id}")

            # Stock & Option buttons in the same row
            col1, col2 = st.columns(2)
            with col1:
                st.button("Stock", key=f"stock_{pos_id}", on_click=select_stock, args=(pos_id,))
            with col2:
                st.button("Option", key=f"option_{pos_id}", on_click=select_option, args=(pos_id,))

            # Check which position type was selected and display corresponding inputs
            selected_type = st.session_state["selected_position"].get(pos_id)

            if selected_type == "Stock":
                col1, col2, col3 = st.columns(3)
                with col1:
                    ticker = st.text_input(f"Stock Ticker {pos_id}", key=f"ticker_{pos_id}")
                with col2:
                    stock_price = st.number_input(f"Stock Price {pos_id}", key=f"stock_price_{pos_id}")
                with col3:
                    quantity = st.number_input(f"Quantity {pos_id}", min_value=1, key=f"quantity_{pos_id}")
                
                # Store in portfolio (optional)
                portfolio.add_stock_position(stock_price, quantity)

            elif selected_type == "Option":
                col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)
                with col1:
                    spot_price = st.number_input(f"Spot ", key=f"spot_{pos_id}")
                with col2:
                    quantity = st.number_input(f"Quantity ", min_value=1, key=f"quantity_{pos_id}")
                with col3:
                    K = st.number_input(f"Strike ", key=f"strike_{pos_id}")
                with col4:
                    sigma = st.number_input(f"Sigma ", key=f"sigma_{pos_id}")
                with col5:
                    T = st.number_input(f"DTE ", key=f"dte_{pos_id}")
                with col6:
                    r = st.number_input(f"r ", key=f"r_{pos_id}")
                with col7:
                    q = st.number_input(f"q ", key=f"q_{pos_id}")
                with col8:
                    otype = st.selectbox(f"Type ", ["Call", "Put"], key=f"otype_{pos_id}")
                with col9:
                    exposure = st.selectbox(f"L/S ", ["Long", "Short"], key=f"L/S_{pos_id}")

                # Store in portfolio (optional)
                portfolio.add_option_position(None, spot_price, K, T, sigma, r, otype, q, quantity)

def calc_payoff(x_values: list[float], x_axis_choice: str, portfolio: Portfolio, model: str, calc_type: str):

    option_model = OptionPriceFactory().create_model(model, calc_type)

    payoff = np.zeros_like(x_values)

    for position in portfolio.positions:
        if repr(position) == "Option":

            if x_axis_choice == "Spot Price":
                position.S = np.array(x_values)
            elif x_axis_choice == "Time to Expiry":
                position.T = np.array(x_values)
            elif x_axis_choice == "Volatility":
                position.sigma = np.array(x_values)

            if position.otype == "Call":
                payoff += option_model.call(position.S, position.K, position.T, position.sigma, position.r, position.q) * position.quantity

            elif position.otype == "Put":
                payoff += option_model.put(position.S, position.K, position.T, position.sigma, position.r, position.q) * position.quantity

        elif position.__repr__() == "Stock":
            payoff += position.price * position.quantity
    
    return payoff

def plot_payoff(x_values: list[float], x_axis_choice: str, portfolio: Portfolio, model_choice: str, calc_type: str):

    y = calc_payoff(x_values, x_axis_choice, portfolio, model_choice, calc_type)

    fig, ax = plt.subplots()
    ax.plot(x_values, y, label="Payoff")
    ax.set_title("Payoff")
    ax.legend()
    st.pyplot(fig)

    return None

def main():
    if "positions" not in st.session_state:
        st.session_state["positions"] = []

    if "selected_position" not in st.session_state:
        st.session_state["selected_position"] = {}

    portfolio = Portfolio()

    st.button("Add Position", on_click=add_position)

    x_axis_choice = st.sidebar.selectbox("Select Change Variable:", ["Spot Price", "Time to Expiry", "Volatility"])
    model_choice = st.sidebar.selectbox("Select Model:", ["Bsm", "Heston", "SABR"])


    # pick model
    if model_choice == "Bsm":
        calc_type = st.sidebar.selectbox("Select Calculation Type:", ["Analytical", "Simulation"])
    elif model_choice == "Heston":
        calc_type = st.sidebar.selectbox("Select Calculation Type:", ["FFT", "Simulation"])
    elif model_choice == "SABR":
        calc_type = st.sidebar.selectbox("Select Calculation Type:", ["Analytical", "Simulation"])

    # pick option price sensitivity change
    if x_axis_choice == "Spot Price":
        min_spot = st.sidebar.number_input("Min Spot Price")
        max_spot = st.sidebar.number_input("Max Spot Price")
        x_values = np.linspace(min_spot, max_spot, 100)
    elif x_axis_choice == "Time to Expiry":
        x_values = st.sidebar.slider("DTEs", 0.01, 3.0, value=(0.01, 3.0))
    elif x_axis_choice == "Volatility":
        x_values = st.sidebar.slider("Volatility", 0.05, 3.0, value=(0.01, 3.0))

    st.sidebar.button("Plot Payoff", on_click= lambda: plot_payoff(x_values, x_axis_choice, portfolio, model_choice, calc_type))

    display_positions(portfolio)

if __name__ == "__main__":
    main()