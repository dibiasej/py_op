import streamlit as st

st.header("Volatility Surface Practice")

col1, col2 = st.columns(2)

def skew_button():
    st.write("Skew")

def term_structure_button():
    st.write("Term Structure")

with col1:

    if st.button("Skew"):  
        skew_button()

with col2:

    if st.button("Term Structure"):
        term_structure_button()