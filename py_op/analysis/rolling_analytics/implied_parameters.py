
class RollingGVV:
    pass

class RollingHeston:
    pass

class RollingSABR:
    pass

class RollingrBergomi:
    pass

class RollingSVI:
    pass

class RollingSkewModel:
    """
    This class will be similar to the RollingSkew class in implied_surface.py except metrics from this will be defined from some model
    """
    def skew_curve(self, parameterization):
        """
        This method gets the whole skew curve over time for a specified model
        parameterization str: either strike, moneyness, log moneyness or delta
        """
        pass

    def implied_skew_moneyness(self):
        pass