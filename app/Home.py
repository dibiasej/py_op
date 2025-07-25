import streamlit as st
import matplotlib.pyplot as plt
import sys
import os

from calc_engine.option_pricing import analytical_solutions as an
from . import optimize as opt

def main_button():
    # Check if `selected_option` exists in session state
    if "selected_option" not in st.session_state:
        st.session_state.selected_option = None

    # Create horizontal buttons using `st.columns()`
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("Calculations")
        if st.button("Optimize"):
            st.session_state.selected_option = "Optimize"

    # Display the current selected option
    st.write(f"Selected Option: {st.session_state.selected_option}")

    return st.session_state.selected_option
        
SCREEN_FACTORY = {'Optimize' : opt.main}

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