import streamlit as st
import matplotlib.pyplot as plt
import sys
import os

from calc_engine.option_pricing import analytical_solutions as an
from app import volatility_surface as vs
from app import realized_volatility as rv
from app import skew as sk

def main_button():
    # Check if `selected_option` exists in session state
    if "selected_volatility_option" not in st.session_state:
        st.session_state.selected_volatility_option = None

    # Create horizontal buttons using `st.columns()`
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("Surface Dynamics")
        if st.button("Volatility Surface"):
            st.session_state.selected_volatility_option = "Volatility Surface"

        if st.button("Skew"):
            st.session_state.selected_volatility_option = "Skew"
            pass

        if st.button("Term Structure"):
            #st.session_state.selected_option = "Term Structure"
            pass

    with col2:
        st.markdown("Rolling Metrics")

        if st.button("Realized Volatility"):
            st.session_state.selected_volatility_option = "Realized Volatility"

    # Display the current selected option
    st.write(f"Selected Option: {st.session_state.selected_volatility_option}")

    return st.session_state.selected_volatility_option
        
SCREEN_FACTORY = {'Volatility Surface' : vs.main,
                  "Realized Volatility" : rv.main,
                  "Skew": sk.main}

def main():

    st.markdown("# Volatility")

    st.session_state['current_screen'] = main_button()

    #try:
    
    display_object = SCREEN_FACTORY[str(st.session_state["current_screen"])]

    display_object()

        #display_object.create_display()

        #display_object.plot_display()
    
    #except:
        #st.write("None")

main()