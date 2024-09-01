import numpy as np
from scipy.stats import norm, ncx2
from scipy.integrate import quad

"""
We might add a class Analytical that is a Factory to create one of our option pricing model
"""

class BachelierAnalytical:

    @staticmethod
    def call(S: float, K: int, T: float, bsm_sigma: float, r: float = .05):
        """
        The Bachelier model is not giving us the same option price as bsm. In order to fully understand this model we need to read the saved pdf using Bachelier for bsm practicioners
        Note: we are now able to get the correct market price, we are suppose to multiply the bsm sigma by the current stock price to get the bachelier sigma
        """
        bach_sigma = bsm_sigma * S
        d = (S * np.exp(r * T) - K) / (bach_sigma * np.sqrt(T))
        return np.exp(-r * T)* bach_sigma * np.sqrt(T) * (d * norm.cdf(d) + norm.pdf(d))
    
    @staticmethod
    def put(S: float, K: int, T: float, bsm_sigma: float, r: float = .05):
        """
        The Bachelier model is not giving us the same option price as bsm. In order to fully understand this model we need to read the saved pdf using Bachelier for bsm practicioners
        Note: we are now able to get the correct market price, we are suppose to multiply the bsm sigma by the current stock price to get the bachelier sigma
        """
        bach_sigma = bsm_sigma * S
        d = (S * np.exp(r * T) - K) / (bach_sigma * np.sqrt(T))
        return np.exp(-r * T)* bach_sigma * np.sqrt(T) * (-d * norm.cdf(-d) + norm.pdf(-d))

class BlackScholesMertonAnalytical:

    @staticmethod
    def put(S: float, K: float, T: float, sigma: float, r: float = 0.05, q: float = 0) -> float:
        d1: float = (np.log(S/K) + (r - q + ((sigma**2)/2)) * T) / (sigma*np.sqrt(T))
        d2: float = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r*T) * norm.cdf(-d2) - S*np.exp(-q*T) * norm.cdf(-d1)

    @staticmethod
    def call(S: int, K: int, T: float, sigma: float, r: float = .06, q: float = 0) -> float:
        d1 = (np.log(S/K) + (r - q + ((sigma**2)/2)) * T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# ncx2 should be a chi squared distribution
class CEVAnalytical:

    """
    Note our vol parameter is different then bsm and depends on the beta parameter
     -- If beta = 1 we get bsm and the sigma parameter to plug in will be decimal ex .25
     -- If beta = 0 we get bachelier and the sigma parameter to plug in will be interger ex 25
    For 0 < beta < 1, the value of what sigma parameter to plug in will be .25 < sigma < 25
    Note: Chris K's book does give a relationship between the cev_call and bsm_call which we may be able to work into these below methods in order to get a call and put method
          that except the bsm sigma as a parameter then transforms it into one that works for the cev formula.
                                                            cev_vol = sigma / (S**beta)
    """

    @staticmethod
    def call(S: float, K: int, T: float, sigma: float, r: float = .05, q: float = 0, beta: float = .99):
        v = 1/(2*(1-beta))
        x_1 = 4*(v**2)*(K**(1/v))/((sigma**2) * T)
        x_2 = 4*(v**2)*((S*np.exp(r*T))**(1/v))/((sigma**2) * T)
        kappa_1 = 2*v + 2
        kappa_2 = 2*v
        lambda_1 = x_2
        lambda_2 = x_1
        return np.exp(-r*T)*((S*np.exp(r*T)*(1-ncx2.cdf(x_1,kappa_1,lambda_1))) - K*ncx2.cdf(x_2,kappa_2,lambda_2))

    @staticmethod
    def put(S: float, K: int, T: float, sigma: float, r: float = .05, q: float = 0, beta: float = .99):
        v = 1/(2*(1-beta))
        x_1 = 4*(v**2)*(K**(1/v))/((sigma**2) * T)
        x_2 = 4*(v**2)*((S*np.exp(r*T))**(1/v))/((sigma**2) * T)
        kappa_1 = 2*v + 2
        kappa_2 = 2*v
        lambda_1 = x_2
        lambda_2 = x_1
        return np.exp(-r*T)*(K * (1 - ncx2.cdf(x_2, kappa_2, lambda_2)) - (S*np.exp(r*T)*(ncx2.cdf(x_1, kappa_1, lambda_1))))

class AnalyticalPriceFactory:
    @staticmethod
    def create_model(model_name):
        match model_name.lower():
            case 'bachelier':
                return BachelierAnalytical
            case 'blackscholesmerton' | "blackscholes" | 'bsm' | 'bs':
                return BlackScholesMertonAnalytical
            case 'constant elasticity variance' | "cev":
                return CEVAnalytical
            case _:
                raise ValueError(f"Unknown model name: {model_name}")

def main():

    """
    Note anything in this main function is purely ran as a test and has/should have no influence with production
    """
    bsm = BlackScholesMertonAnalytical()
    bach = BachelierAnalytical()

    print(f"bsm {bsm.call(552, 555, 1/12, .14)}")
    print(f"bach {bach.call(552, 555, 1/12, .14)}")

    return None

if __name__ == "__main__":
    print(main())