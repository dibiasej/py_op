import numpy as np

def call_expiration_price(spot: np.array, strike: int, exposure: str = "long"):

    return np.maximum(spot - strike, 0) if exposure == "long" else -np.maximum(spot - strike, 0)

def put_expiration_price(spot: np.array, strike: int, exposure: str = "long"):

    return np.maximum(strike - spot, 0) if exposure == "long" else -np.maximum(strike - spot, 0)

def main():

    spot = np.linspace(10, 100, 100)

    return call_expiration_price(spot, 50, "short")

if __name__ =="__main__":
    print(main())