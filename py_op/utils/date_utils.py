import datetime as dt
import numpy as np

"""
As of now these date utils will be a bunch of functions but in the future we want to create a more robust and organized way to use these utils
"""

MARKET_HOLIDAYS: list[str] = ["2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19", "2024-07-04", "2024-08-02", "2024-11-28", "2024-12-25", "2025-01-01", "2025-01-09", "2025-01-20", '2025-02-17']
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
    Turns a string specifying when the option maturity is into a int representing the number of days, ex '1M' turns into 30.
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
    
def fetch_exp(o_graph, mat_days: int, max_diff_days: int = 4):
    """
    Given target maturity in DAYS (e.g., 30), return the expiration date (YYYY-MM-DD)
    whose DTE is closest, but only if it's within max_diff_days.

    Returns:
      - None  (if no expiration is close enough)
      - (exp_str, diff_days, actual_dte_days) otherwise
    """
    exp_list = list(o_graph.get_common_exps())

    exp_list.sort()
    
    target = np.timedelta64(int(mat_days), "D")
    max_diff = np.timedelta64(int(max_diff_days), "D")

    first, last = 0, len(exp_list) - 1
    closest_exp = None
    closest_diff = None
    closest_actual = None

    close_date = np.datetime64(o_graph.close_date)

    while first <= last:
        mid = (first + last) // 2

        exp_dt = np.datetime64(exp_list[mid])
        actual = exp_dt - close_date
        diff = abs(actual - target)

        if closest_diff is None or diff < closest_diff:
            closest_diff = diff
            closest_exp = exp_list[mid]
            closest_actual = actual

        if actual < target:
            first = mid + 1
        elif actual > target:
            last = mid - 1
        else:
            closest_diff = np.timedelta64(0, "D")
            closest_exp = exp_list[mid]
            closest_actual = actual
            break

    if closest_diff is None or closest_diff > max_diff:
        # Not close enough
        # optional: return diagnostic info instead of None
        return ValueError(f"DTE is not close enough to expirations it must be within {max_diff_days} days")

    diff_days = int(closest_diff / np.timedelta64(1, "D"))
    actual_days = int(closest_actual / np.timedelta64(1, "D"))
    return closest_exp, diff_days, actual_days

def find_bracketing_dtes(dte_list, target_dte):
    """
    This function returs the closes two dtes between target_dte, we pass in dte_list and a target_dte for ex 30 and it returns the two closes ex (28, 34)
    Returns (dte_lo, dte_hi) where:
      - if exact match exists: (target_dte, None)
      - else: (nearest below, nearest above)
      - if target outside range: (nearest end, None)  [you can choose to return (None, None) instead]
    Assumes dte_list is sorted ascending.
    """
    if not dte_list:
        return None, None

    # exact match
    if target_dte in dte_list:
        return target_dte, None

    # outside range -> clamp to nearest available
    if target_dte < dte_list[0]:
        return dte_list[0], None
    if target_dte > dte_list[-1]:
        return dte_list[-1], None

    # bracket inside range
    for lo, hi in zip(dte_list[:-1], dte_list[1:]):
        if lo < target_dte < hi:
            return lo, hi

    return None, None