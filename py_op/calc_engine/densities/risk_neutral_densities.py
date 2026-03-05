import numpy as np

from calc_engine.option_pricing.analytical_solutions import BlackScholesMertonAnalytical

def breenden_litzenberger(prices, strikes, dte, r):
    """
    The function calculates the Implied density given market price. This works for unequal distant strikes so if the three strikes are 100, 105, 108 it still works.
    If the probability is less than 0 we do not append it to our distribution and exclude it from the data
    Important notes:
    - This works best if we pass in mid prices on equal space nodes 
    - The dte we pass is must be divided by 365
    """

    probs, new_strikes = [], []

    for i in range(1, len(prices) - 1):
        h1 = strikes[i] - strikes[i - 1]
        h2 = strikes[i + 1] - strikes[i]

        probability = np.exp(r * dte) * (2.0 / (h1 + h2)) * (
            (prices[i + 1] - prices[i]) / h2
            - (prices[i] - prices[i - 1]) / h1
        )

        if probability > 0:
            probs.append(probability)
            new_strikes.append(strikes[i])

    return probs, new_strikes

def breeden_litzenberger_ivs(S, ivs, strikes, dte, r = 0.04, q = 0, otype = "call"):
    """
    This function takes ivs instead of prices, then uses black scholes to convert to prices then derives the implied density
    We can use something like a gvv interpolation to get a lot of ivs between non existant strikes, then plug the interpolated ivs and strikes
    into this function to get a very smooth distribution.

    In the future we are going to add functionality for other models like Heston and SABR.
    """
    ivs = np.array(ivs)
    strikes = np.array(strikes)

    bsm = BlackScholesMertonAnalytical()

    if otype == "call":
        prices = bsm.call(S, strikes, dte, ivs, r, q)

    elif otype == "put":
        prices = bsm.put(S, strikes, dte, ivs, r, q)

    return breenden_litzenberger(prices, strikes, r)