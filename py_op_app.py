import streamlit as st

st.set_page_config(
    page_title="Deboption Research",
    page_icon="📈",
)

from app import Home

if __name__ == "__main__":
    Home.main()