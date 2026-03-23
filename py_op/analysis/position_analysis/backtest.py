
from py_op.data.structures.position_info_data_structure import PortfolioInfo, StockPositionInfo, OptionPositionInfo
from py_op.data.builders.position_series_builder import PositionSeriesBuilder

class Backtest:

    def __init__(self, portfolio: PortfolioInfo, position_series_builder = PositionSeriesBuilder()):
        self.portfolio: PortfolioInfo = portfolio
        self.position_series_builder = position_series_builder

    def add_stock(self):
        pass