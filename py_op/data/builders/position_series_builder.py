import numpy as np

#from data.price_data import get_close_prices
from py_op.data.structures.position_info_data_structure import PortfolioInfo, StockPositionInfo, OptionPositionInfo
from py_op.data.repositories.position_repository import BacktestRepository

class PositionSeriesBuilder:

    def __init__(self, portfolio: PortfolioInfo, repo = BacktestRepository()) -> None:
        self.portfolio: PortfolioInfo = portfolio
        self.repo = repo
        self.portfolio_data = {}

    def get_positions(self):

        start_date = self.portfolio.start_date
        end_date = self.portfolio.end_date

        for position in self.portfolio:
            #print(f"position: {position}")

            ticker = position.ticker
            exposure = position.exposure

            if isinstance(position, StockPositionInfo):

                #close_prices = get_close_prices(ticker, start_date, end_date)
                spot_dates, spot_prices = zip(*self.repo.get_stock_price_history(ticker, start_date, end_date))
                spot_dates, spot_prices = np.array(spot_dates), np.array(spot_prices)
                self.portfolio_data[(ticker, "stock")] = (spot_prices * position.quantity, spot_dates) if exposure == "long" else (-spot_prices * position.quantity, spot_dates)

            elif isinstance(position, OptionPositionInfo):

                otype = position.otype
                exp = position.exp
                
                if position.strike is not None:
                    strike = position.strike
                    # print(f"exp: {exp}")
                    # print(f"start date: {start_date}")

                    # we still need to make it times 100 for notional value
                    #print(f"ticker = {ticker}, expiration={exp}, strike={strike}, option_type={otype}, start_date={start_date}, end_date={end_date}")
                    rows = self.repo.get_option_price_history_by_strike(ticker = ticker, expiration = exp, strike = strike, option_type=otype, start_date=start_date, end_date=end_date)
                    #print(f"rows: {rows}")
                    close_dates, mid_prices, strikes, dtes, close_prices, spot_prices = zip(*rows)
                    mid_prices = np.array(mid_prices)
                    self.portfolio_data[(ticker, otype, strike, exp)] = (mid_prices * position.quantity, close_dates) if exposure == "long" else (-mid_prices * position.quantity, close_dates)

                elif position.moneyness is not None:

                    moneyness = position.moneyness
                    #print(f"ticker = {ticker}, expiration={exp}, moneyness={moneyness}, option_type={otype}, start_date={start_date}, end_date={end_date}")
                    rows = self.repo.get_option_price_history_by_moneyness(ticker = ticker, expiration=exp, moneyness=moneyness, option_type=otype, start_date=start_date, end_date=end_date)
                    close_dates, mid_prices, strikes, dtes, close_prices, spot_prices = zip(*rows)
                    self.portfolio_data[(ticker, otype, moneyness, exp)] = (mid_prices * position.quantity, close_dates) if exposure == "long" else (-mid_prices * position.quantity, close_dates)
        
        return self.portfolio_data