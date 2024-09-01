from calc_engine.option_pricing import simulation as sim
from calc_engine.option_pricing import analytical_solutions as an
from calc_engine.implied_volatility import iv_calc as iv
from option_data.process_option_chain import OptionFactory

import matplotlib.pyplot as plt

def main():

    spy_graph = OptionFactory().create_option_graph("SPY")

    spy_skew = spy_graph.get_skew("2024-08-30")

    print(spy_skew[0].get_price())

    bsm_sim_model = sim.BlackScholesSimulation()

    bsm_ananlytical_model = an.BlackScholesMertonAnalytical()

    heston_model = sim.HestonSimulation()

    iv_bsm_ob = iv.ImpliedVolatility(bsm_sim_model)

    iv_heston_ob = iv.ImpliedVolatility(heston_model)

    #iv_vol_bsm = iv_bsm_ob.bisection_method(9.10, 541, 545, 1/12)

    #print(f"iv bsm root {imp_vol_bsm_root}")

    #print(f"iv heston root {imp_vol_heston_root}")

    #print(f"iv bsm newton: {iv_vol_bsm}")

    print(f"bsm call {bsm_ananlytical_model.call(541, 545, 1/12, .156, .05)}\n")

    for option in spy_skew:
        market_price = option.get_price()
        strike = option.get_strike()
        if strike < 500 and strike > 580:
            continue
        dte = option.get_dte()
        print(f"dte {dte}")

        iv_vol_bsm = iv_bsm_ob.bisection_method(market_price, 541, strike, dte/252)
        print(f"iv {iv_vol_bsm}")
        print(f"strike {strike}")
        print(f"market price {market_price}")
        print(f"\n")

    return None

if __name__ == "__main__":
    print(main())