

from option_data.process_option_chain import OptionFactory

def price_surface(ticker: str, close_date: str, option_type: str):
    o_graph = OptionFactory().create_option_graph(ticker, close_date, option_type)
    price_surface = []
    for 