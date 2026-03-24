import numpy as np

#from data.price_data import get_close_prices
from py_op.data.structures.position_info_data_structure import PortfolioInfo, StockPositionInfo, OptionPositionInfo
from py_op.data.repositories.position_repository import PositionSeriesRepository
from py_op.calc_engine.vol_engine.iv_calc import RootFinder
from py_op.calc_engine.greeks.analytical_greeks import AnalyticalDelta

class PositionSeriesBuilder:

    def __init__(self, portfolio: PortfolioInfo, repo = PositionSeriesRepository()) -> None:
        self.portfolio: PortfolioInfo = portfolio
        self.repo = repo
        self.portfolio_data = {}

    def _delta_hedge_helper(self, market_prices, strikes, dtes, spot_prices, otype, r = 0.04, q = 0):
        """
        Note right now we only hedge using iv which is an issue because we may want to hedge with rv
        """
        deltas = []
        for i in range(len(market_prices)):
            iv = RootFinder().calculate(market_prices[i], spot_prices[i], strikes[i], dtes[i]/365, r=r, otype=otype, q=q)
            delta = AnalyticalDelta().calculate(spot_prices[i], strikes[i], dtes[i]/365, iv, r=r, q=q, otype=otype)
            deltas.append(delta)

        return np.array(deltas)*100

    def get_positions(self, r: float = 0.04, q: float = 0):

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
                    mid_prices = np.array(mid_prices)
                    self.portfolio_data[(ticker, otype, moneyness, exp)] = (mid_prices * position.quantity, close_dates) if exposure == "long" else (-mid_prices * position.quantity, close_dates)
        
                elif position.delta is not None:
                    delta = position.delta
                    data = self.repo.get_option_price_history_by_delta(ticker = ticker, expiration=exp, delta=delta, option_type=otype, start_date=start_date, end_date=end_date, r=r, q=q)
                    close_dates, mid_prices, strikes, dtes, spot_prices = zip(*data)
                    mid_prices = np.array(mid_prices)
                    self.portfolio_data[(ticker, otype, delta, exp)] = (mid_prices * position.quantity, close_dates) if exposure == "long" else (-mid_prices * position.quantity, close_dates)

                if position.delta_hedged == True:

                    deltas = self._delta_hedge_helper(mid_prices, strikes, dtes, spot_prices, otype, r=r, q=q)

                    if position.exposure == "long" and otype == "put":
                        delta_hedge = -deltas*spot_prices

                    elif position.exposure == "long" and otype == "call":
                        print("in long call")
                        delta_hedge = -deltas*spot_prices

                    else:
                        delta_hedge = deltas*spot_prices
                    
                    #print(f"delta hedge: {delta_hedge}")
                    print(f"delta hedge: {delta_hedge}")

                    self.portfolio_data[(ticker, "delta hedge", None, exp)] = delta_hedge
                    print(f"deltas: {deltas}")
                    print(f"spot: {spot_prices}")
                    print("\n")


        return self.portfolio_data