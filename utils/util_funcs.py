import datetime as dt

MARKET_HOLIDAYS: list[str] = ["2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19", "2024-07-04", "2024-08-02", "2024-11-28", "2024-12-25"]
MISSED_DAYS: list[str] = ["2024-01-04"]

def option_date_range(start: str, end: str) -> list[str]:
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