import numpy as np
from abc import ABC, abstractmethod

from ..skew import skew_filter as sp

"""
I think x_data should be a list of strikes
"""

class SkewFit:

    @abstractmethod
    def fit(self, implied_volatilities: list[float], x_data: list[float]):
        pass

class StickyStrike(SkewFit):

    def fit(self, implied_volatilities: list[float], x_data: list[float], dte: float) -> (list[float], list[float]):

        if dte > 1:
            dte /= 252

        X = np.array([[1, x, x**2, dte, dte**2, x*dte] for x in x_data])
        b_hat, residuals, rank, s = np.linalg.lstsq(X, implied_volatilities, rcond=None)
        ivs = X @ b_hat

        return ivs, x_data
    
class RelativeStickyDelta(SkewFit):

    # This is the relative sticky delta model implementation according to John C Hull's paper https://www-2.rotman.utoronto.ca/~hull/DownloadablePublications/DaglishHullSuoRevised.pdf

    def fit(self, implied_volatilities: list[float], x_data: list[float], dte: float, S: float, filter_logic: str = None) -> (list[float], list[float]):

        if dte > 1:
            dte /= 252

        F: float = S * np.exp(.01 * dte)

        moneyness_filter = sp.LogMoneyness()

        if filter_logic is None:

            ivs, strikes, moneyness_data = moneyness_filter.filter(implied_volatilities, x_data, S = F)

        else:

            ivs, strikes, moneyness_data = moneyness_filter.filter(implied_volatilities, x_data, filter_logic = filter_logic, S = F)

        X = np.array([[1, moneyness, moneyness**2, dte, dte**2, moneyness*dte] for moneyness in moneyness_data])
        #X = np.array([[1, np.log(x / F), np.log(x / F)**2, dte, dte**2, np.log(x / F)*dte] for x in x_data])
        b_hat, residuals, rank, s = np.linalg.lstsq(X, ivs, rcond=None)
        ivs_regressed = X @ b_hat

        return ivs_regressed, strikes, moneyness_data
    
class StationarySquareRootTime:

    def fit(self, implied_volatilities: list[float], x_data: list[float], dte: float, S: float, filter_logic: str = None) -> (list[float], list[float]):

        if dte > 1:
            dte /= 252

        F: float = S * np.exp(.01 * dte)

        moneyness_time_filter = sp.LogMoneynessTime()

        if filter_logic is None:

            ivs, strikes, moneyness_time_data = moneyness_time_filter.filter(implied_volatilities, x_data, dte = dte, S = F)

        else:

            ivs, strikes, moneyness_time_data = moneyness_time_filter.filter(implied_volatilities, x_data, dte = dte, filter_logic = filter_logic, S = F)

        X = np.array([[1, moneyness, moneyness**2, dte, dte**2, moneyness*dte] for moneyness in moneyness_time_data])
        #X = np.array([[1, np.log(x / F), np.log(x / F)**2, dte, dte**2, np.log(x / F)*dte] for x in x_data])
        b_hat, residuals, rank, s = np.linalg.lstsq(X, ivs, rcond=None)
        ivs_regressed = X @ b_hat

        return ivs_regressed, strikes, moneyness_time_data
    
class StationarySquareRootTimePoly:

    def fit(self, implied_volatilities: list[float], x_data: list[float], dte: float, S: float, filter_logic: str = None) -> (list[float], list[float]):

        if dte > 1:
            dte /= 252

        F: float = S * np.exp(.01 * dte)

        moneyness_time_filter = sp.LogMoneynessTime()

        if filter_logic is None:

            ivs, strikes, moneyness_time_data = moneyness_time_filter.filter(implied_volatilities, x_data, dte = dte, S = F)

        else:

            ivs, strikes, moneyness_time_data = moneyness_time_filter.filter(implied_volatilities, x_data, dte = dte, filter_logic = filter_logic, S = F)

        X = np.array([[1, moneyness, moneyness**2, moneyness**3, moneyness**4] for moneyness in moneyness_time_data])
        #X = np.array([[1, np.log(x / F), np.log(x / F)**2, dte, dte**2, np.log(x / F)*dte] for x in x_data])
        b_hat, residuals, rank, s = np.linalg.lstsq(X, ivs, rcond=None)
        ivs_regressed = X @ b_hat

        return ivs_regressed, strikes, moneyness_time_data
    
class Polynomial(SkewFit):

    def fit(self):
        pass