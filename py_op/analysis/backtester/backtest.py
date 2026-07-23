import numpy as np

from py_op.data.structures.position_info_data_structure import PortfolioInfo, StockPositionInfo, OptionPositionInfo
from py_op.data.builders.position_series_builder import PositionSeriesBuilder

class Backtest:

    def __init__(self, start_date: str, end_date: str):
        self.portfolio: PortfolioInfo = PortfolioInfo(start_date, end_date)

    def add_option(self, ticker: str, exp: str, strike: int = None, moneyness: float = None, delta: float = None, exposure: str = "long", otype: str = "call", quantity: int = 1, delta_hedged: bool = False) -> None:
        self.portfolio.add_option(ticker = ticker, exp = exp, strike = strike, moneyness = moneyness, delta = delta, exposure = exposure, otype = otype, quantity = quantity, delta_hedged = delta_hedged)

    def add_stock(self, ticker: str, exposure: str = "long", quantity: int = 1) -> None:
        self.portfolio.add_stock(ticker = ticker, exposure = exposure, quantity = quantity)

    def calculate_position_value(self):
        position_series = PositionSeriesBuilder(self.portfolio)
        portfolio_data = position_series.get_positions()
        pnl = 0
        for key, value in portfolio_data.items():

            if key[1] == "put" or key[1] == "call":
                option_pos = np.array(value[0]) * 100
                pnl += option_pos

            else:
                pnl += np.array(value[0])

        return pnl
    
    def calculate_pnl(self):
        position_value = self.calculate_position_value()
        pnl = position_value - position_value[0]
        return pnl