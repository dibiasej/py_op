import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import option_data as od
from skew_toolset import skew_fit as sk

TICKER = st.text_input("Enter a Ticker")
EXPRIRATION = st.text_input("Enter the Option Expiration ('YYYY-MM-DD')")
MODEL = st.text_input("Enter the Model")

stock = yf.Ticker(TICKER)

# Get the current price
CURRENT_PRICE = stock.history(period="1d")['Close'][0]

def get_data(ticker):
    o_factory = od.OptionFactory()
    o_graph = o_factory.create_option_graph(ticker, "2024-10-08")
    return o_graph

OPTION_GRAPH = get_data(TICKER)


def sticky_strike_skew(exp: str):
    skew = OPTION_GRAPH.get_skew(exp)
    sticky_strike = sk.StickyStrike()
    return sticky_strike.fit(skew.implied_volatilities(), skew.strikes(), skew.get_dte())

def relative_sticky_delta_skew(exp: str):
    skew = OPTION_GRAPH.get_skew(exp)
    sticky_strike = sk.RelativeStickyDelta()
    return sticky_strike.fit(skew.implied_volatilities(), skew.strikes(), skew.get_dte(), CURRENT_PRICE)

def market_skew(exp: str):
    skew = OPTION_GRAPH.get_skew(exp)
    return skew

if MODEL == "Sticky Strike":

    ivs, strikes = sticky_strike_skew(EXPRIRATION)

    fig, ax = plt.subplots()
    ax.plot(strikes, ivs)

    # Add labels
    ax.set_xlabel("Strike Prices")
    ax.set_ylabel("Implied Volatilities")
    ax.set_title(f"{TICKER} IV Skew\n{EXPRIRATION}")

    st.pyplot(fig)

elif MODEL == "Relative Sticky Delta":

    ivs, strikes, moneyness = relative_sticky_delta_skew(EXPRIRATION)

    fig, ax = plt.subplots()
    ax.plot(moneyness, ivs)

    # Add labels
    ax.set_xlabel("Moneyness")
    ax.set_ylabel("Implied Volatilities")
    ax.set_title(f"{TICKER} IV Skew\n{EXPRIRATION}")

    st.pyplot(fig)

else:
    skew = market_skew(EXPRIRATION)
    fig, ax = plt.subplots()
    ax.plot(skew.strikes(), skew.implied_volatilities())

    ax.set_xlabel("Strike Prices")
    ax.set_ylabel("Implied Volatilities")
    ax.set_title(f"{TICKER} IV Skew\n{EXPRIRATION}")

    #skew_fig = sticky_strike_skew(EXPRIRATION)
    # Display the Matplot`lib figure in Streamlit
    st.pyplot(fig)