import streamlit as st

from data.option_data.load_option_chain import TODAY
from data.data_processor.data_processor import MarketPriceProcessor
from data.data_processor.skew_processor import ConstructSkew
from data.price_data import process_price_data
from utils import date_utils

def main():

    ticker = str(st.sidebar.text_input("Enter a Ticker"))

    close_date = str(st.sidebar.date_input("Close Date"))

    market_processor = MarketPriceProcessor(ticker, close_date)
    exps = market_processor.option_call_graph.get_expirations()
    exp = st.selectbox("Select Exp:", options=exps)

    model_name = str(st.selectbox("Select Model:", options=['Heston', 'SABR']))

    ConstructSkew(model_name)
    
    #start_date = str(st.sidebar.date_input("Enter the Start Date ('YYYY-MM-DD')"))
    #end_date = str(st.sidebar.date_input("Enter the End Date ('YYYY-MM-DD')"))

    #close_dates = date_utils.option_close_date_range(start_date, end_date)
    #S_list, S_dates = process_price_data.get_close_prices(ticker, start_date, end_date)
    #common_dates = [date for date in close_dates if date in S_dates]

    #skew_dates = [end_date]

    #selected_close_date = st.sidebar.selectbox("Common Dates", common_dates)

    #market_processors = [MarketPriceProcessor(ticker, date) for date in skew_dates]

    #expir = {date: MarketPriceProcessor(ticker, date).option_call_graph.get_expirations() for date in skew_dates}
    #st.write("exp", expir)