import numpy as np
from numpy.polynomial.polynomial import Polynomial

from data.option_data.process_option_chain import OptionFactory
from calc_engine.volatility import iv_calc
from calc_engine.volatility import local_volatility as lv
from calc_engine.option_pricing import analytical_solutions as an

def market_price_generator(ticker: str, close_date: str, strikes: int, expirations=None, option_type: str = 'call'):
    """
    Generator that yields OptionNode objects based on specified expirations and strikes.

    :param option_type: Type of option ('call' or 'put')
    :param expirations: List of expiration dates to process
    :param strikes: List of strikes to process
    :yield: OptionNode object
    """
    option_graph = OptionFactory().create_option_graph(ticker, close_date, option_type=option_type)
    
    if expirations is None:
        expirations = list(option_graph.get_expirations())

    atm_option = option_graph.get_atm_option(expirations[1], close_date)
    atm_strike = atm_option.get_strike()
    atm_iv = atm_option.get_implied_volatility()

    for exp in expirations:
        price_list = []
        new_strikes = []
        for strike in strikes:

            if (exp, strike) not in option_graph.nodes:
                continue

            option = option_graph.nodes[(exp, strike)]

            market_price = option.get_price()
            dte = option.get_dte()
            new_strikes.append(strike)
            price_list.append(market_price)

        yield (price_list, dte, new_strikes, atm_strike, atm_iv)

def market_iv_generator(ticker: str, close_date: str, strikes: int, expirations=None, option_type: str = 'call'):
    """
    Generator that yields OptionNode objects based on specified expirations and strikes.

    :param option_type: Type of option ('call' or 'put')
    :param expirations: List of expiration dates to process
    :param strikes: List of strikes to process
    :yield: OptionNode object
    """
    option_graph = OptionFactory().create_option_graph(ticker, close_date, option_type=option_type)
    
    if expirations is None:
        expirations = list(option_graph.get_expirations())

    atm_option = option_graph.get_atm_option(expirations[1], close_date)
    atm_strike = atm_option.get_strike()

    for exp in expirations:
        iv_list = []
        new_strikes = []
        for strike in strikes:

            if (exp, strike) not in option_graph.nodes:
                continue

            option = option_graph.nodes[(exp, strike)]

            market_iv = option.get_implied_volatility()
            market_price = option.get_price()
            dte = option.get_dte()
            new_strikes.append(strike)

            iv_list.append(market_iv)

        yield (iv_list, dte, new_strikes, atm_strike)

def bsm_iv_generator(ticker: str, close_date: str, strikes: int, expirations=None, model = an.BlackScholesMertonAnalytical(), option_type: str = 'call'):
    """
    Generator that yields OptionNode objects based on specified expirations and strikes.

    :param option_type: Type of option ('call' or 'put')
    :param expirations: List of expiration dates to process
    :param strikes: List of strikes to process
    :yield: OptionNode object
    """
    option_graph = OptionFactory().create_option_graph(ticker, close_date, option_type=option_type)
    iv_calc_obj = iv_calc.ImpliedVolatility()
    
    if expirations is None:
        expirations = list(option_graph.get_expirations())

    atm_option = option_graph.get_atm_option(expirations[1], close_date)
    atm_strike = atm_option.get_strike()

    for exp in expirations:
        iv_list = []
        new_strikes = []
        for strike in strikes:

            if (exp, strike) not in option_graph.nodes:
                continue

            option = option_graph.nodes[(exp, strike)]

            market_iv = option.get_implied_volatility()
            market_price = option.get_price()
            dte = option.get_dte()
            new_strikes.append(strike)

            bs_iv = iv_calc_obj.newtons_method(market_price, atm_strike, strike, dte, 0, otype=option_type)

            if bs_iv == 0 or bs_iv is None:
                iv_list.append(market_iv * 100)
            else:
                iv_list.append(bs_iv)
                    
        yield (iv_list, dte, new_strikes)

def local_volatility_generator(ticker: str, close_date: str, strikes: int, expirations=None, option_type: str = 'call'):
    option_graph = OptionFactory().create_option_graph(ticker, close_date, option_type=option_type)
    lv_calc_obj = lv.LocalVolatility()
    
    if expirations is None:
        expirations = list(option_graph.get_expirations())

    atm_option = option_graph.get_atm_option(expirations[1])
    atm_strike = atm_option.get_strike()

    for exp in expirations:
        local_vol_list = []
        new_strikes = []

        for strike in strikes:

            if (exp, strike) not in option_graph.nodes:
                continue

            option = option_graph.nodes[(exp, strike)]

            market_iv = option.get_implied_volatility()
            market_price = option.get_price()
            dte = option.get_dte()
            new_strikes.append(strike)

            local_vol = lv_calc_obj.dupire_finite_difference(atm_strike, strike, dte, market_iv, otype=option_type)

            if local_vol == 0 or local_vol == None:
                local_vol_list.append(market_iv * 100)
            else:
                local_vol_list.append(local_vol * 100)
        
        yield (local_vol_list, dte, new_strikes)

def local_volatility_surface_data_processor(ticker: str, close_date: str, expirations: list[str], strikes: list[float], option_type: str):
    local_vol_obj = lv.LocalVolatility()
    o_graph = OptionFactory().create_option_graph(ticker, close_date, option_type = option_type)
    dtes = []
    price_surface = []

    print(f"o graph {o_graph}")

    for exp in expirations:
        skew = o_graph.get_skew(exp)
        dte = skew.get_dte()
        dtes.append(dte)
        price_list = []
        new_strikes = []

        for option in skew:
            if option.get_strike() in strikes:
                price_list.append(option.get_price())
                new_strikes.append(option.get_strike())

        print(f"strikes {len(strikes)}")
        print(f"price list {len(price_list)}\n")
        new_strikes = np.array(new_strikes)
        p = Polynomial.fit(np.array(new_strikes), np.array(price_list), deg=3)
        x_poly = np.linspace(new_strikes.min(), new_strikes.max(), 100)
        y_poly = p(x_poly)
        
        price_surface.append(y_poly)
    
    price_surface = np.array(price_surface)
    dtes = np.array(dtes) / 252

    loc_vol_surface = local_vol_obj.dupire_price_surface(x_poly, dtes, price_surface)

    return loc_vol_surface, x_poly, dtes

def sabr_generator(ticker: str, close_date: str, strikes: int, expirations=None, model = an.BlackScholesMertonAnalytical(), option_type: str = 'call'):
    option_graph = OptionFactory().create_option_graph(ticker, close_date, option_type=option_type)
    iv_calc_obj = iv_calc.ImpliedVolatility(model=model)
    
    if expirations is None:
        expirations = list(option_graph.get_expirations())

    atm_option = option_graph.get_atm_option(expirations[1], close_date)
    atm_strike = atm_option.get_strike()

    for exp in expirations:
        iv_list = []
        new_strikes = []
        for strike in strikes:

            if (exp, strike) not in option_graph.nodes:
                continue

            option = option_graph.nodes[(exp, strike)]

            market_iv = option.get_implied_volatility()
            market_price = option.get_price()
            dte = option.get_dte()
            new_strikes.append(strike)

            bs_iv = iv_calc_obj.newtons_method(market_price, atm_strike, strike, dte, 0, otype=option_type)

            if bs_iv == 0 or bs_iv is None:
                iv_list.append(market_iv * 100)
            else:
                iv_list.append(bs_iv)
                    
        yield (iv_list, dte, new_strikes)

def price_surface_heston(ticker: str, close_date: str, strikes: list[int], expirations: list[str] = None, option_type: str = 'call'):
    o_graph = OptionFactory().create_option_graph(ticker, close_date, option_type=option_type)

    if expirations is None:
        expirations = list(o_graph.get_expirations())

    dte_list = []
    price_surf = []
    strike_data_list = strikes

    #atm_option = o_graph.get_atm_option(expirations[1], close_date)
    #atm_strike = atm_option.get_strike()
    #tm_iv = atm_option.get_implied_volatility()

    for exp in expirations:
        price_list = []
        indices_to_remove = []  # Collect indices to remove
        skew = o_graph.get_skew(exp)
        dte = skew.get_dte()
        dte_list.append(dte)


        # Identify indices to remove
        for idx, strike in enumerate(strike_data_list):
            if (exp, strike) not in o_graph.nodes:
                indices_to_remove.append(idx)
            else:
                option = o_graph.nodes[(exp, strike)]
                price_list.append(option.get_price())

        # Remove items in reverse order to avoid index shifting
        for idx in sorted(indices_to_remove, reverse=True):
            for p_list in price_surf:
                p_list.pop(idx)
            strike_data_list.pop(idx)

        price_surf.append(price_list)

    return price_surf, dte_list, strike_data_list, atm_strike, atm_iv