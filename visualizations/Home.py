import streamlit as st
import matplotlib.pyplot as plt
import sys
import os

from calc_engine.option_pricing import analytical_solutions as an
from . import volatility_surface as vs
from . import gamma_exposure as ge
from . import realized_volatility as rv
from . import risk_measures as rm
from . import back_test as bt

def main_button():
    # Check if `selected_option` exists in session state
    if "selected_option" not in st.session_state:
        st.session_state.selected_option = None

    # Create horizontal buttons using `st.columns()`
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("Volatility")
        if st.button("Volatility Surface"):
            st.session_state.selected_option = "Volatility Surface"

        if st.button("Realized Volatility"):
            st.session_state.selected_option = "Realized Volatility"

    with col2:
        st.markdown("Position Analysis")
        if st.button("Back Testing"):
            st.session_state.selected_option = "Back Testing"

        if st.button("Risk Measures"):
            st.session_state.selected_option = "Risk Measures"

    with col3:
        st.markdown("Analysis Tools")
        if st.button("Gamma Exposure"):
            st.session_state.selected_option = "Gamma Exposure"

    # Display the current selected option
    st.write(f"Selected Option: {st.session_state.selected_option}")

    return st.session_state.selected_option
        
SCREEN_FACTORY = {'Back Testing' : bt.main,
                  'Volatility Surface' : vs.main,
                  'Gamma Exposure' : ge.main,
                  "Realized Volatility" : rv.main,
                  "Risk Measures" : rm.main}

def main():

    st.markdown("# Option Research Platform")

    st.session_state['current_screen'] = main_button()

    #try:
    
    display_object = SCREEN_FACTORY[str(st.session_state["current_screen"])]

    display_object()

        #display_object.create_display()

        #display_object.plot_display()
    
    #except:
        #st.write("None")

#main()