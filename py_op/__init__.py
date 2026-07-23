
# Include oprion_price_factory eventually

"""
Below is my old imports but it is very sloppy and I want to remake it
The import would look like
py_op.analytical_solutions.SABRAnalytical()  -- This is behavior I dont want
Instead we should import the factory method
"""

from .calc_engine import option_pricing
from .calc_engine import vol_engine as volatility_models
from .analysis import rolling_analytics


# from py_op.calc_engine.option_pricing import (
#     analytical_solutions,
#     FFT,
#     fourier_inversion,
#     simulation,
# )

# from py_op.calc_engine.calibration import model_optimization
# from py_op.calc_engine.greeks import analytical_greeks

# # IV + Skew tools (class-level exports)
# from py_op.calc_engine.vol_engine.iv_calc import (
#     ImpliedVolatility,
#     RootFinder,
#     BisectionMethod,
#     NewtonsMethod,
#     InverseGaussian,
#     SkewCalculator,
#     TermStructureCalculator,
# )

from py_op.calc_engine.densities import risk_neutral_densities

from py_op.data.price_data import process_price_data

"""
Below is for direct imports, so instead of having to dig down through modules and packages to get something we can directly call our code from py_op
"""
from py_op.data.builders.option_chain_builder import create_chain, create_chain_series
from py_op.calc_engine.vol_engine.iv_calc import ImpliedVolatility as implied_volatility_calculator


# for analytics I am thinking I want it to look like py_op.analytics.RollingTermStructure() in a notebook
from . import analysis
from py_op.analysis.backtester import backtest

__all__ = [
    # Pricing modules
    "analytical_solutions",
    "FFT",
    "fourier_inversion",
    "simulation",
    # IV / Skew classes
    "ImpliedVolatility",
    "RootFinder",
    "BisectionMethod",
    "NewtonsMethod",
    "SkewCalculator",
    "TermStructureCalculator",
    "models",
    # Data
    "process_price_data",
    "create_chain", 
    "create_chain_series",
    "BacktestBuilder",
    "PortfolioInfo",
    # analytics
    "analysis",
    "backtest",
    "rolling_analytics"
]

__version__ = "0.1.0"