import numpy as np

from data.data_processor.data_processor import ModelPriceProcessor, VolatilityProcessor
from calc_engine.option_pricing.option_price_factory import OptionPriceFactory
from calc_engine.volatility.gvv import GVV



"""
!!! I think as of 7/25/25 I will get rid of this module, or at least use more design patterns for it
Notes: We will need to use MarketPriceProcessor to get market data first in jupyter notebook
Note: We will need to calibrate some of the models first

Idea: Might need to use an adapter for non similar models 

"""

# Models
def bsm(S, market_prices, strikes, dte):
    strikes, ivs = zip(*VolatilityProcessor().otm_put_call_skew(S, market_prices, strikes, dte))
    return strikes, ivs

def sabr(S, strikes, dte, sigma_0, alpha, beta, rho, calculation_type = "analytical"):
    sabr_model = OptionPriceFactory().create_model("sabr", calculation_type)
    model_strikes, model_prices = zip(*ModelPriceProcessor(sabr_model).put_call_price_skew(S, strikes, dte, sigma_0=sigma_0, alpha=alpha, beta=beta, rho=rho))
    _, ivs = zip(*VolatilityProcessor().otm_put_call_skew(S, model_prices, model_strikes, dte))
    return model_strikes, ivs

def heston(S, strikes, dte, v0, theta, kappa, sigma, rho, calculation_type = "FFT"):
    heston_model = OptionPriceFactory().create_model('heston', calculation_type)
    model_strikes, model_prices = zip(*ModelPriceProcessor(heston_model).put_call_price_skew(S, strikes, dte, v0=v0, theta=theta, kappa=kappa, sigma=sigma, rho=rho))
    _, ivs = zip(*VolatilityProcessor().otm_put_call_skew(S, model_prices, model_strikes, dte))
    return model_strikes, ivs

def rbergomi(S, strikes, dte, xi, a, rho, eta, calculation_type = "sim"):
    rbergomi_model = OptionPriceFactory().create_model('rbergomi', calculation_type)
    model_strikes, model_prices = zip(*ModelPriceProcessor(rbergomi_model).put_call_price_skew(S, strikes, dte, xi=xi, a=a, rho=rho, eta=eta))
    _, ivs = zip(*VolatilityProcessor().otm_put_call_skew(S, model_prices, model_strikes, dte))
    return model_strikes, ivs

def merton_jump_diffusion(S, strikes, dte, mu, sig, lam, muJ, sigJ, calculation_type = "FFT"):
    merton_jump_diffusion_model = OptionPriceFactory().create_model("merton jump", calculation_type)
    model_strikes, model_prices = zip(*ModelPriceProcessor(merton_jump_diffusion_model).put_call_price_skew(S, strikes, dte, mu=mu, sig=sig, lam=lam, muJ=muJ, sigJ=sigJ))
    _, ivs = zip(*VolatilityProcessor().otm_put_call_skew(S, model_prices, model_strikes, dte))
    return model_strikes, ivs


# Fitting
def sticky_strike(strikes, ivs, dte):
    X = np.array([[1, strike, strike**2, dte, dte**2, strike*dte] for strike in strikes])
    b_hat, residuals, rank, s = np.linalg.lstsq(X, ivs, rcond=None)
    regressed_ivs = X @ b_hat

    return regressed_ivs

def relative_sticky_delta(S, strikes, ivs, dte, r = 0.04):
    F = S*np.exp(r*dte)
    X = np.array([[1, np.log(strike / F), np.log(strike / F)**2, dte, dte**2, np.log(strike / F)*dte] for strike in strikes])
    b_hat, residuals, rank, s = np.linalg.lstsq(X, ivs, rcond=None)
    regressed_ivs = X @ b_hat
    return regressed_ivs

def stationary_square_root_time(S, strikes, ivs, atm_iv, dte, r = 0.04):
    F = S*np.exp(r*dte)
    #y = np.array(ivs) - atm_iv
    X = np.array([[atm_iv, np.log(strike / F)/np.sqrt(dte), (np.log(strike / F)/np.sqrt(dte))**2, (np.log(strike / F)/np.sqrt(dte))**3] for strike in strikes])
    b_hat, residuals, rank, s = np.linalg.lstsq(X, ivs, rcond=None)
    regressed_ivs = X @ b_hat
    return regressed_ivs   