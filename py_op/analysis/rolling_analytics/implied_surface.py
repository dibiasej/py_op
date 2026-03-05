
"""
(3/1/2026) We will move the skew and constant maturity IV's from implied_values.py into here
"""

class RollingTermStructure:
    """
    This class will be used to get rolling term structure analytics
    First we will make it compatible with the z-score analytics from Euan sinclairs book
    Also make it compatible with rolling dte2 - dte1 where dte2 > dte1
        Either add argument to each method that says z_score = True, or make completely seperate methods eg, atmf_zscore
    We may make something like this called RollingSkew as well   
    """

    def atmf():
        pass

    def constant_maturity():
        pass

    def variance_swap():
        pass


class RollingSkew:
    pass

class RollingKurtosis:
    pass

class RollingIV:
    pass