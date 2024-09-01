import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import datetime as dt

from rvol_helper_functions import _t
from rvol_helper_functions import _length
from rvol_helper_functions import _get_resampled_prices
from rvol_helper_functions import _get_log_rets
from rvol_helper_functions import _dates


"""
Note currently we have a class for Realized volatility that calculates close to close but we want to make a class for each individual realized volatility
calculation ie Garman Klass class, Parkinson class, etc.. that have then all become objects using a match case type of frame work in a Realized volatility class. 
"""

class RealizedVolatility:

    def __init__(self, ticker: str, start: str, end: str, rvol_length: str = "Month", sampling_freq: str = "D") -> None:

        self._ticker: str = ticker
        self._start: str = start
        self._end: str = end
        self._length: str = rvol_length
        self._sampling_freq: str = sampling_freq
        self._recalculate()

    def _recalculate(self):
        self._prices: pd.DataFrame = _get_resampled_prices(self._ticker, self._start, self._end, self._sampling_freq)
        self._log_rets: np.ndarray = _get_log_rets(self._prices)
        #self._dates: list = _dates(self._prices, self._length, self._sampling_freq)

    def set_length(self, length):
        
        self._length = length
        self._recalculate()

    def set_sampling_freq(self, sampling_freq):

        self._sampling_freq = sampling_freq
        self._recalculate()

    def set_length_freq(self, length, sampling_freq):

        self._length = length
        self._sampling_freq = sampling_freq
        self._recalculate()

    def npclose_to_close_rvol(self, length: int = "Month", sampling_freq: str = "D"):

        if length != "Month" or sampling_freq != "D":

            if isinstance(length, int) and sampling_freq == "D":

                t: int = _t(self._sampling_freq)
                length_int = length

            else:

                self.set_length_freq(length, sampling_freq)

                t: int = _t(self._sampling_freq)
                length_int: int = _length(self._length, self._sampling_freq)

        else:

            t: int = _t(self._sampling_freq)
            length_int: int = _length(self._length, self._sampling_freq)
            
        rVol: list = [np.sqrt(sum(self._log_rets[0 + i: length_int + i]**2) / length_int) * np.sqrt(t) for i in range(len(self._log_rets) - length_int)]

        return rVol
    
    def min_max_avg_plot(self, weeks: int = None):
    
        start: dt.datetime = dt.datetime.strptime(self._start, "%Y-%m-%d")
        end: dt.datetime = dt.datetime.strptime(self._end, "%Y-%m-%d")
        days = (end - start).days
        maxRvol: list = []
        avgRvol: list = []
        minRvol: list = []
        x: list = []
        for week, i in enumerate(range(4, days, 5)):

            if week == weeks:
                break
            rvol = self.npclose_to_close_rvol(i)

            if len(rvol) == 0:
                break
            
            x.append(i)
            maxRvol.append(np.amax(rvol))
            avgRvol.append(np.mean(rvol))
            minRvol.append(np.amin(rvol))

        fig, ax = plt.subplots()
        ax.grid(True)
        line1, = ax.plot(x, maxRvol)
        line2, = ax.plot(x, minRvol)
        line3, = ax.plot(x, avgRvol)
        ax.set_ylabel("Volatility")
        ax.set_xlabel("X Week Realized Volaility")
        plt.xticks(rotation = 45)
        plt.show()
    
    def plot(self, length: int = "Month", sampling_freq: str = "D"):

        rvol: np.ndarray = self.npclose_to_close_rvol(length, sampling_freq)
        dates = _dates(self._prices, length, sampling_freq)
        plt.plot(dates, rvol)
        plt.title(f"{length} Realized Volatiltiy")
        plt.xlabel("Dates")
        plt.ylabel("Volatility")
        plt.xticks(rotation = 45)
        plt.show()



if __name__ == "__main__":
        rvol = RealizedVolatility("spy", "2011-1-1", "2018-11-23")

        rvol.npclose_to_close_rvol()