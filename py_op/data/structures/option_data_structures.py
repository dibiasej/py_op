import numpy as np

from py_op.utils.date_utils import fetch_exp

class OptionContract:
    """
    Node Data structure representing an option Contract.
    """

    def __init__(self, spot: float, strike: float, price: float, bid: float, ask: float, yahoo_iv: float, exp: str, dte: int, volume: float, open_interest: float, otype: str, iv: float = None, delta: float = None) -> None:
        self.spot = spot
        self.strike = strike
        self.price = price
        self.mid_price = (bid + ask) / 2 if bid is not None and ask is not None else price
        self.bid = bid
        self.ask = ask
        self.moneyness = (strike / spot)
        self.log_moneyness = np.log(strike / spot)
        self.yahoo_iv = yahoo_iv
        self.exp = exp
        self.dte = dte
        self.volume = volume
        self.open_interest = open_interest
        self.otype = otype

        # computed analytics (Optional)
        self.iv = iv
        self.delta = delta


    def is_otm(self) -> bool:
        return (
            self.otype == "call" and self.strike > self.spot
        ) or (
            self.otype == "put" and self.strike < self.spot
        )
    
    def __str__(self):
        return "OptionContract"
    
class OptionChain:
    """
    Graph data structure representing an option graph.
    This chain will contain both puts and calls so we wont need to make to seperate graphs.
    """
    def __init__(self):
        self.option_nodes = {}
        self.skew_data = {}
        self.term_structure_data = {}
        self.ticker: str = None
        self.close_date: str = None
        self.S: float = None

    def add_option(self, option_contract: OptionContract) -> None:

        key = (option_contract.exp, option_contract.strike, option_contract.otype)

        if key in self.option_nodes:
            return 

        self.option_nodes[key] = option_contract
        self.term_structure_data.setdefault(option_contract.strike, []).append(key)
        self.skew_data.setdefault(option_contract.exp, []).append(key)

    def get_option(self, strike: int, otype: str, exp: str = None, dte: int = None, max_days_diff: int = 4) -> OptionContract:

        if dte is not None:
            exp_data = self.get_exp_from_dte(dte, max_days_diff)
            exp = exp_data[0]
        elif dte is None and exp is None:
            return ValueError("DTE or expiration must be defines")
        
        strikes_at_exp = np.array([strike[1] for strike in self.skew_data[exp] if strike[2] == otype])

        option_idx = np.abs(strikes_at_exp - strike).argmin()
        new_strike = strikes_at_exp[option_idx]

        return self.option_nodes.get((exp, new_strike, otype))

    def get_common_strikes(self):
        """
        Retrieve strikes that are common to both puts and calls. Possibly for a certain expiration 
        If we are doing anything across expirations/dte ie term structure work this method will give an error because this gets every possible
        strike in the chain. We might get a strike from dte = 30 that isnt in dte = 60
        """
        call_strikes = []
        put_strikes = []

        for key, value in self.option_nodes.items():

            if value.otype == "call":
                call_strikes.append(value.strike)

            elif value.otype == "put":
                put_strikes.append(value.strike)

        return sorted(set(put_strikes) & set(call_strikes))
    
    def get_common_strikes_at_exp(self, exp: str = None, dte: int = None, max_diff_days = 4) -> list[float]:

        if dte is not None:
            exp_data = self.get_exp_from_dte(dte, max_diff_days=max_diff_days)            
            exp = exp_data[0]
            data = self.skew_data[exp]

            actual_dte = exp_data[2]

            call_strikes = {strike for exp, strike, opt in data if opt == "call"}
            put_strikes = {strike for exp, strike, opt in data if opt == "put"}
            return sorted(call_strikes & put_strikes), actual_dte

        elif dte is None and exp is None:
            return ValueError("exp or dte must be defined both cannot be None")

        data = self.skew_data[exp]

        call_strikes = {strike for exp, strike, opt in data if opt == "call"}
        put_strikes = {strike for exp, strike, opt in data if opt == "put"}
        return sorted(call_strikes & put_strikes)

    def get_common_exps(self):
        """
        Retrieve expirations that are common to both puts and calls. Possibly for a certain strike
        """
        call_exps = []
        put_exps = []

        for key, value in self.option_nodes.items():

            if value.otype == "call":
                call_exps.append(value.exp)

            elif value.otype == "put":
                put_exps.append(value.exp)

        return sorted(set(put_exps) & set(call_exps))
    
    def get_common_dtes(self):
        """
        Retrieve dtes that are common to both puts and calls. Possibly for a certain strike
        """
        call_dtes = []
        put_dtes = []

        for key, value in self.option_nodes.items():

            if value.otype == "call":
                call_dtes.append(value.dte)

            elif value.otype == "put":
                put_dtes.append(value.dte)

        return sorted(set(put_dtes) & set(call_dtes))
    
    def get_exp_from_dte(self, mat_days: int, max_diff_days: int = 4):
        return fetch_exp(self, mat_days, max_diff_days)
    
    def get_call_skew(self, exp: str = None, dte: int = None, max_days_diff: int = 4):
        """
        This methods fetches all option contract for calls across strike for one expiration.
        """
        if dte is not None:
            exp_data = self.get_exp_from_dte(dte, max_days_diff)
            exp = exp_data[0]
        elif dte is None and exp is None:
            raise ValueError("DTE or expiration must be defines")
        
        return [self.option_nodes[key] for key in self.skew_data[exp] if self.option_nodes[key].otype == "call"]

    def get_put_skew(self, exp: str = None, dte: int = None, max_days_diff: int = 4):
        """
        This methods fetches all option contract for puts across strike for one expiration.
        """
        if dte is not None:
            exp_data = self.get_exp_from_dte(dte, max_days_diff)
            exp = exp_data[0]
        elif dte is None and exp is None:
            raise ValueError("DTE or expiration must be defines")
        
        return [self.option_nodes[key] for key in self.skew_data[exp] if self.option_nodes[key].otype == "put"]

    def get_otm_skew(self, exp: str = None, dte: int = None, max_days_diff: int = 4):
        """
        This method fetches otm puts and otm calls
        """
        if dte is not None:
            exp_data = self.get_exp_from_dte(dte, max_days_diff)
            exp = exp_data[0]
        elif dte is None and exp is None:
            raise ValueError("DTE or expiration must be defines")
        
        puts = [self.option_nodes[key] for key in self.skew_data[exp] if self.option_nodes[key].otype == "put" and self.option_nodes[key].is_otm()]
        calls = [self.option_nodes[key] for key in self.skew_data[exp] if self.option_nodes[key].otype == "call" and self.option_nodes[key].is_otm()]
        return puts + calls
    
    def get_equal_skew(self, exp: str = None, dte: int = None, max_days_diff: int = 4):
        """
        Return list of (put_contract, call_contract) tuples for strikes that exist in BOTH puts and calls
        for the given expiration (or closest expiration to the given DTE).
        """
        if dte is not None:
            exp_data = self.get_exp_from_dte(dte, max_days_diff)
            if exp_data is None:
                return []  # or raise ValueError
            exp = exp_data[0]
            
        elif exp is None:
            raise ValueError("DTE or expiration must be defined")

        keys = self.skew_data.get(exp, [])
        if not keys:
            return []

        puts_by_strike = {}
        calls_by_strike = {}

        for key in keys:
            c = self.option_nodes[key]
            # normalize strike if you need (optional):
            # strike = round(float(c.strike), 4)
            strike = c.strike

            if c.otype == "put":
                puts_by_strike[strike] = c
            elif c.otype == "call":
                calls_by_strike[strike] = c

        common_strikes = sorted(puts_by_strike.keys() & calls_by_strike.keys())

        # list of tuples: (put, call)
        return [(puts_by_strike[k], calls_by_strike[k]) for k in common_strikes]

    def get_equal_term_structure(self, strike):
        """
        This method returns a list of tuples where each tuple has a put and a call OptionContract in it.
        The list is ordered by expiration starting from the earliest going to the latest. 
        We can pass in a strike to get a specific term structure at a certain strike
        """
        pairs = []
        for exp in sorted({k[0] for k in self.term_structure_data[strike]}):
            put_key, call_key = (exp, strike, "put"), (exp, strike, "call")
            if put_key in self.option_nodes and call_key in self.option_nodes:
                pairs.append((self.option_nodes[put_key], self.option_nodes[call_key]))

        return pairs

    def get_equal_surface(self, strikes: list[float] = None, exps: list[str] = None):
        """
        This method can return a price surface in either 
            1) term structure direction
            2) Skew direction 

        We need to correct this function so that each list inside the list has equal len
        """
        if strikes is not None and exps is not None:
            return ValueError("One of the arguments, strikes or exps, has to be None")
        
        option_surface = []

        if strikes is not None:
            for strike in strikes:
                term_structure = self.get_equal_term_structure(strike)
                option_surface.append(term_structure)

            return option_surface
        
        elif exps is not None:
            for exp in exps:
                skew = self.get_equal_skew(exp)
                option_surface.append(skew)

            return option_surface
        
    def get_call_prices(self, exp: str = None, dte: int = None, max_days_diff: int = 4, mid_price = True):

        options = self.get_call_skew(exp=exp, dte=dte, max_days_diff=max_days_diff)

        if mid_price == True:

            prices, strikes, dtes = zip(*[(option.mid_price, option.strike, option.dte) for option in options])
        
        else:

            prices, strikes, dtes = zip(*[(option.price, option.strike, option.dte) for option in options])

        return prices, strikes, dtes[0]
    
    def get_put_prices(self, exp: str = None, dte: int = None, max_days_diff: int = 4, mid_price = True):

        options = self.get_put_skew(exp=exp, dte=dte, max_days_diff=max_days_diff)

        if mid_price == True:

            prices, strikes, dtes = zip(*[(option.mid_price, option.strike, option.dte) for option in options])
        
        else:

            prices, strikes, dtes = zip(*[(option.price, option.strike, option.dte) for option in options])

        return prices, strikes, dtes[0]
        
    def get_otm_skew_prices(self, exp: str = None, dte: int = None, max_days_diff: int = 4, mid_price = True):
        """
        Returns: 
        prices: otm option prices [put_price.., call_price]
        strikes:
        moneyness: (might add log moneyness)
        dtes: actual dtes which differ slightly by the one we passed in
        Arguments:
        exp: str representing opition expiration YYYY-MM-DD
        dte: in representing days till expiration
        max_days_diff: the maximum amount of days our dte is allowed to stray from what is actually on the option chain.
                        Ex: if our dte = 30 and the option chains closes dte to that is 36 we return an error.
        """
        otm_options = self.get_otm_skew(exp=exp, dte=dte, max_days_diff=max_days_diff)

        if mid_price == True:

            prices, strikes, dtes = zip(*[(option.mid_price, option.strike, option.dte) for option in otm_options])

            zero_frac = sum(p == 0 for p in prices) / len(prices)
            if zero_frac > .5:

                prices, strikes, dtes = zip(*[(option.price, option.strike, option.dte) for option in otm_options])
        
        else:

            prices, strikes, dtes = zip(*[(option.price, option.strike, option.dte) for option in otm_options])

        return prices, strikes, dtes[0]
    
    def get_equal_skew_prices(self, exp: str = None, dte: int = None, max_days_diff: int = 4, mid_price = True):
        put_call_options = self.get_equal_skew(exp, dte, max_days_diff)

        if mid_price == True:

            put_prices, call_prices, strikes, dtes = zip(*[(put.mid_price, call.mid_price, put.strike, put.dte) for put, call in put_call_options])
            
            zero_threshold = 0.5  # 80% zeros means unusable

            put_zero_frac = sum(p == 0 for p in put_prices) / len(put_prices)
            call_zero_frac = sum(c == 0 for c in call_prices) / len(call_prices)

            if put_zero_frac > zero_threshold or call_zero_frac > zero_threshold:
                put_prices, call_prices, strikes, dtes = zip(
                    *[(put.price, call.price, put.strike, put.dte)
                    for put, call in put_call_options]
                )

        else:

            put_prices, call_prices, strikes, dtes = zip(*[(put.price, call.price, put.strike, put.dte) for put, call in put_call_options])

        return put_prices, call_prices, strikes, dtes
    
    def get_equal_term_structure_prices(self, strike, mid_price = True):
        options_term_structure = self.get_equal_term_structure(strike)

        if mid_price == True:

            put_prices, call_prices, dtes = zip(*[(put.mid_price, call.mid_price, put.dte) for put, call, in options_term_structure])

        else:

            put_prices, call_prices, dtes = zip(*[(put.price, call.price, put.dte) for put, call, in options_term_structure])

        return put_prices, call_prices, dtes
    
    def get_equal_term_structure_atf_prices(self, dtes: list[int] = None, r: float = 0.04, mid_price: bool = True, max_diff_days: int = 4):
        """
        This method calculates a list of forward prices for every expiry, then it fetches the corresponding atf strike for every expiry and the correspoding put and call prices.
        The beauty of this method is no matter what for a single expiration and strike we get put and call prices so there will never be a miss match (ie put price when there is no call, etc).
        Usually we should not pass in any arguments into this function but if you want to get prices at specific dtes, use OptionChain.get_common_dtes() then make a list of dtes from that and pass it in as a argument.
        Returns:
        put_prices: list of atf put prices
        call_prices: list of atf call prices
        final_dtes: list of the dtes for the corresponding option prices
        for_strikes: Forward strikes
        """

        if dtes is None:
            dtes = self.get_common_dtes()

        new_dtes = np.array(dtes) / 365
        S = self.S
        F = S*np.exp(new_dtes*r)

        term_struct_data = []

        for f, dte in zip(F, dtes):
            common_strikes, actual_dte = self.get_common_strikes_at_exp(dte = dte, max_diff_days=max_diff_days)

            option_idx = np.abs(common_strikes - f).argmin()
            strike_at_forward = common_strikes[option_idx]
            term_struct = self.get_equal_term_structure(strike_at_forward)
            for put, call in term_struct:

                if call.dte == actual_dte and put.dte == actual_dte and call.strike == strike_at_forward and put.strike == strike_at_forward:

                    if mid_price == True:

                        term_struct_data.append((put.mid_price, call.mid_price, call.dte, call.strike))
                    
                    else:

                        term_struct_data.append((put.price, call.price, call.dte, call.strike))
                        
        put_prices, call_prices, final_dtes, for_strikes = zip(*term_struct_data)

        return put_prices, call_prices, final_dtes, for_strikes
    
    def get_equal_price_surface_across_strikes(self, dtes: list[float] = None, exps: list[str] = None, max_days_diff: int = 4):
        """
        This method is used to get both call and put prices for the whole surface across strikes not term structure.
        So given a list of expirations or dtes we get the corresponding put and call prices and their strikes.
        Return:
        data: A dictionary where the key is the expiration and the value is a tuple of three lists put and call prices and strikes
        """
        data = {}

        if dtes is not None:
            for dte in dtes:
                put_prices, call_prices, strikes, dte_list = self.get_equal_skew_prices(dte=dte, max_days_diff=max_days_diff)
                
                data[dte_list[0]] = (put_prices, call_prices, strikes)

        elif exps is not None:
            for exp in exps:
                put_prices, call_prices, strikes, dte_list = self.get_equal_skew_prices(exp=exp)
                data[dte_list[0]] = (put_prices, call_prices, strikes)

        return data

    def get_otm_price_surface_across_strikes(self, dtes: list[float] = None, exps: list[str] = None, max_days_diff: int = 4):
        data = {}

        if dtes is not None:
            for dte in dtes:
                prices, strikes, dte_ = self.get_otm_skew_prices(dte=dte, max_days_diff=max_days_diff)
                data[dte_] = (prices, strikes)

        elif exps is not None:
            for exp in exps:
                prices, strikes, dte_ = self.get_otm_skew_prices(exp=exp)
                data[dte_] = (prices, strikes)

        return data

    def __str__(self):
        return "OptionChain"