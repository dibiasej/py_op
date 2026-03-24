class StockPositionInfo:
    """
    This is a class to hold meta data for a position.
    """
    def __init__(self, ticker, exposure: str, quantity: int = 1) -> None:
        self.ticker: str = ticker
        self.exposure: str = exposure
        self.quantity: int = quantity

    def __repr__(self) -> str:
        return f"StockPosition({self.ticker}, {self.exposure}, {self.quantity})"
    
class OptionPositionInfo:
    """
    This is a class to hold meta data for a position.
    """
    def __init__(self, ticker: str, exp:str, strike: int = None, moneyness: float = None, delta: float = None, exposure: str = "long", otype: str = None, quantity: int = 1, delta_hedged: bool = False) -> None:
        self.ticker: str = ticker
        self.exp: str = exp
        self.strike: int = strike
        self.moneyness: int = moneyness
        self.delta: float = delta
        self.exposure: str = exposure
        self.otype: str = otype
        self.quantity: int = quantity
        self.delta_hedged = delta_hedged

    def __repr__(self) -> str:
        return f"OptionPosition({self.ticker}, {self.exposure}, {self.quantity}, {self.otype})"


class PortfolioInfo:
    """
    This will be used to hold position data. We will use this from the client side to define out positions, and pass it to something else that will fetch the positions.
    I want to rename this
    """

    def __init__(self, start_date: str, end_date: str) -> None:
        self.start_date: str = start_date
        self.end_date: str = end_date
        self.positions: list = []

    def add_option(self, ticker: str, exp: str, strike: int = None, moneyness: float = None, delta: float = None, exposure: str = "long", otype: str = "call", quantity: int = 1, delta_hedged: bool = False) -> None:
        position = OptionPositionInfo(ticker = ticker, exp = exp, strike = strike, moneyness=moneyness, delta = delta, exposure=exposure, otype=otype, quantity=quantity, delta_hedged=delta_hedged)
        self.positions.append(position)

    def add_stock(self, ticker: str, exposure: str = "long", quantity: int = 1) -> None:
        position = StockPositionInfo(ticker = ticker, exposure=exposure, quantity=quantity)
        self.positions.append(position)

    def __iter__(self):
        for position in self.positions:
            yield position