import numpy as np

from py_op.data.builders.option_chain_builder import create_chain_series
from py_op.calc_engine.vol_engine.iv_calc import RootFinder, InverseGaussian


class RollingAnalytics:

    def __init__(self, ticker: str, start_date: str, end_date:str, moneyness: float = None, steps: int = 1, iv_calc = InverseGaussian()) -> None:
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.moneyness = moneyness
        self.steps = steps
        self.iv_calc = iv_calc
        self.chain_series = create_chain_series(ticker, start_date, end_date, moneyness=moneyness, steps=steps)