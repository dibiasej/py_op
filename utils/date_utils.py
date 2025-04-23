import datetime as dt
import numpy as np

"""
As of now these date utils will be a bunch of functions but in the future we want to create a more robust and organized way to use these utils
"""

MARKET_HOLIDAYS: list[str] = ["2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19", "2024-07-04", "2024-08-02", "2024-11-28", "2024-12-25"]
MISSED_DAYS: list[str] = ["2024-01-04", "2024-09-02", "2024-10-14", "2024-11-01"]

def create_date_range(start: str, end: str, length: int):
    """
    Creates a date range between a start date and end date for a specific defined length.
    Useful for plotting when the date range is unkown bu the start and end date are known.
    Ex: Useful for plotting and Realized vol measure
    """

    start_date = np.datetime64(start)
    end_date = np.datetime64(end)

    num_points = length

    step = (end_date - start_date) / (num_points - 1)

    dates = np.array([start_date + i * step for i in range(num_points)], dtype="datetime64[D]")

    return dates

def option_close_date_range(start: str, end: str) -> list[str]:
    """
    Utility function for getting a list of close date ranges for our options data 
    """
    start_date = dt.datetime.strptime(start, "%Y-%m-%d")
    end_date = dt.datetime.strptime(end, "%Y-%m-%d")
    numdays: int = (end_date - start_date).days + 1

    date_list = [(start_date + dt.timedelta(days=x-1)).strftime("%Y-%m-%d") for x in range(numdays)
                if (start_date + dt.timedelta(days=x-1)).weekday() < 5
                and (start_date + dt.timedelta(days=x - 1)).strftime("%Y-%m-%d") not in MARKET_HOLIDAYS
                and (start_date + dt.timedelta(days=x - 1)).strftime("%Y-%m-%d") not in MISSED_DAYS
                  ]

    return date_list

def price_date_range(start: str, end: str) -> list[str]:
    start_date = dt.datetime.strptime(start, "%Y-%m-%d")
    end_date = dt.datetime.strptime(end, "%Y-%m-%d")
    numdays: int = (end_date - start_date).days + 1

    date_list = [(start_date + dt.timedelta(days=x-1)).strftime("%Y-%m-%d") for x in range(numdays)
                if (start_date + dt.timedelta(days=x-1)).weekday() < 5
                and (start_date + dt.timedelta(days=x - 1)).strftime("%Y-%m-%d") not in MARKET_HOLIDAYS]
    
    return date_list

def option_exp_to_days(option_exp: str) -> int:
    """
    Turns a string specifying when the option maturity is into a int representing the number of days, ex '1M' turns itno 30.
    """
    num = int(option_exp[:-1])
    unit = option_exp[-1].upper()
    
    if unit == "M":
        return num * 30
    elif unit == "Y":
        return num * 365
    elif unit == "W":
        return num * 7
    elif unit == "D":
        return num
    else:
        raise ValueError(f"Unsupported duration unit: {unit}") 
    
def fetch_exp(o_graph, mat: str | int = "1M"):
    """

    Utility function used with our OptionGraph.
    Given a string or int that represents the time of option maturity ex: 1M, 2M, 30, 60 etc... return the expiration date that most closely matches that.
    """

    first = 0
    last = len(o_graph.get_expirations()) - 1

    if isinstance(mat, str):

        target_mat_days = option_exp_to_days(mat)

    else:
        target_mat_days = mat

    exp_list = list(o_graph.get_expirations())

    closest_exp = None
    closest_diff = np.datetime64("2100-01-01") - np.datetime64("2020-01-01") # originally set as arbitrarily long date

    while first <= last:

        mid = (first + last)//2
        mat_days = np.datetime64(exp_list[mid]) - np.datetime64(o_graph.close_date)


        diff = abs(mat_days - target_mat_days)
        
        if diff < closest_diff:
            
            closest_exp = exp_list[mid]
            closest_diff = diff

        if mat_days == target_mat_days:
            return exp_list[mid]
        
        else:
            if mat_days < target_mat_days:
                first = mid + 1

            elif mat_days > target_mat_days:
                last = mid - 1
        
        curr_mid = mid

    return closest_exp