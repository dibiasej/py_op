import utils
from option_data import OptionFactory

def historical_option_prices(ticker: str, K: int, start_date: str, end_date: str, expiration: str, option_style: str = "european", option_type: str = "call") -> list[float]:

    close_dates: list[str] = utils.option_date_range(start_date, end_date)

    meta_option_price_list: list[int] = []

    for close_date in close_dates:
        option_price: float = OptionFactory().create_option(ticker, K, expiration, close_date, option_style, option_type).get_price()
        meta_option_price_list.append(option_price)

    return meta_option_price_list, close_dates