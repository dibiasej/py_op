from .load_option_chain import SingleUpload

cache: dict = {}

def cache_chain(ticker: str, date: str):

    # This function loads our option chain from our data directory
    # First we check if the chain has already been cached and grab it if it has, and if it has not been cached we load the chain from our data directory and cache it

    if (ticker, date) in cache:
        #print(f"chain already in cache")
        
        return cache[(ticker, date)]
    else:
        #print(f"Adding chain to cache")
        chain = SingleUpload().upload_chain(ticker, date)

        cache[(ticker, date)] = chain

        return chain 

def call_chain_exp(ticker: str, date: str) -> list[str]:

    """List all expiration dates on the option chain. Note the date we pass in as a argument is not the expiration date it is the date of when the close prices were realized"""

    chain = cache_chain(ticker, date)

    return chain[0].keys()

def put_chain_exp(ticker: str, date: str) -> list[str]:
    
    """List all expiration dates on the option chain. Note the date we pass in as a argument is not the expiration date it is the date of when the close prices were realized"""

    chain = cache_chain(ticker, date)

    return chain[1].keys()

def call_exp_strikes(ticker: str, date: str, expiration: str) -> list[int]:

    chain = cache_chain(ticker, date)

    call_chain = chain[0]

    chain_at_exp = call_chain[expiration]

    return chain_at_exp['Strike']

def put_exp_strikes(ticker: str, date: str, expiration: str) -> list[int]:

    chain = cache_chain(ticker, date)

    put_chain = chain[1]
    chain_at_exp = put_chain[expiration]

    return chain_at_exp['Strike']