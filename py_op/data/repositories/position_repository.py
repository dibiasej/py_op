import numpy as np

from py_op.global_variables import OPTION_DB_DIR
from py_op.utils.db_utils import get_connection
from py_op.calc_engine.greeks.analytical_greeks import AnalyticalDelta
from py_op.calc_engine.vol_engine.iv_calc import SkewCalculator, RootFinder

class PositionSeriesRepository:

    def __init__(self, iv_calculator = RootFinder()):
        self.conn = get_connection(OPTION_DB_DIR)
        self.iv_calculator = iv_calculator

    def _validate_dates(self, expiration: str, start_date: str | None, end_date: str | None) -> None:
        if end_date is not None and end_date > expiration:
            raise ValueError(f"end_date ({end_date}) cannot be after expiration ({expiration}).")
        if start_date is not None and end_date is not None and start_date > end_date:
            raise ValueError(f"start_date ({start_date}) cannot be after end_date ({end_date}).")

    def get_option_price_history_by_strike(self, ticker: str, expiration: str, strike: float, option_type: str = "call", start_date: str | None = None, end_date: str | None = None):
        
        self._validate_dates(expiration, start_date, end_date)

        query = """
        SELECT o.close_date, o.mid_price, o.strike, o.dte, o.price, s.close
        FROM option_data o
        JOIN spot_prices s
        ON s.ticker = o.ticker
        AND s.close_date = o.close_date
        WHERE o.ticker = ?
        AND o.expiration_date = ?
        AND o.strike = ?
        AND o.option_type = ? 
        """
        params = [ticker, expiration, strike, option_type]

        if start_date is not None:
            query += " AND o.close_date >= ?"
            params.append(start_date)

        if end_date is not None:
            query += " AND o.close_date <= ?"
            params.append(end_date)

        query += " ORDER BY o.close_date"

        cur = self.conn.cursor()
        cur.execute(query, params)
        return cur.fetchall()
    
    def get_option_price_history_by_moneyness(self, ticker: str, expiration: str, moneyness: float, option_type: str = "call", start_date: str | None = None, end_date: str | None = None):
        self._validate_dates(expiration, start_date, end_date)

        query = """
        WITH ranked AS (
            SELECT
                o.close_date,
                o.mid_price,
                CAST(o.strike AS REAL) AS strike,
                o.dte,
                o.price,
                CAST(s.close AS REAL) AS spot,
                ROW_NUMBER() OVER (
                    PARTITION BY o.close_date
                    ORDER BY ABS((CAST(o.strike AS REAL) / CAST(s.close AS REAL)) - ?) ASC
                ) AS rn
            FROM option_data o
            JOIN spot_prices s
            ON s.ticker = o.ticker
            AND s.close_date = o.close_date
            WHERE o.ticker = ?
            AND o.expiration_date = ?
            AND o.option_type = ?
            AND s.close > 0
        """

        params = [float(moneyness), ticker, expiration, option_type]

        if start_date is not None:
            query += " AND o.close_date >= ?"
            params.append(start_date)

        if end_date is not None:
            query += " AND o.close_date <= ?"
            params.append(end_date)

        query += """
        )
        SELECT close_date, mid_price, strike, dte, price, spot
        FROM ranked
        WHERE rn = 1
        ORDER BY close_date;
        """

        cur = self.conn.cursor()
        cur.execute(query, params)
        return cur.fetchall()

    def get_option_price_history_across_expiration(self, ticker: str, expiration: str, option_type: str = "call", start_date: str | None = None, end_date: str | None = None):
        """
        This method gets a time series of option prices at a specific maturity for multiple strikes (ie skew)
        """
        self._validate_dates(expiration, start_date, end_date)

        query = """
        SELECT o.close_date, o.mid_price, o.strike, o.dte, o.price, s.close
        FROM option_data o
        JOIN spot_prices s
            ON s.ticker = o.ticker
            AND s.close_date = o.close_date
        WHERE o.ticker = ?
        AND o.expiration_date = ?
        AND o.option_type = ?
        """
        params = [ticker, expiration, option_type]

        if start_date is not None:
            query += " AND o.close_date >= ?"
            params.append(start_date)

        if end_date is not None:
            query += " AND o.close_date <= ?"
            params.append(end_date)

        query += " ORDER BY o.close_date, o.strike"

        cur = self.conn.cursor()
        cur.execute(query, params)
        return cur.fetchall()
    
    def get_option_price_history_by_delta(self, ticker: str, expiration: str, delta: float, option_type: str = "call", start_date: str = None, end_date: str = None, r = 0.04, q = 0):
        rows = self.get_option_price_history_across_expiration(ticker = ticker, expiration = expiration, option_type = option_type, start_date = start_date, end_date = end_date)
        pos_data = {}
        skew_calc = SkewCalculator(self.iv_calculator)

        for close_date, mid_price, strike, dte, close_price, spot_price in rows:
            if close_date not in pos_data:
                pos_data[close_date] = {
                    "strikes": [],
                    "mid_prices": [],
                    "dtes": [],
                    "close_prices": [],
                    "spot_prices": []
                }

            pos_data[close_date]["strikes"].append(strike)
            pos_data[close_date]["mid_prices"].append(mid_price)
            pos_data[close_date]["dtes"].append(dte)
            pos_data[close_date]["close_prices"].append(close_price)
            pos_data[close_date]["spot_prices"].append(spot_price)

        data = []
        for key, value in pos_data.items():
            strikes = np.array(value["strikes"])
            mid_prices = np.array(value["mid_prices"])
            dte = value["dtes"][0]
            spot = value["spot_prices"][0]
            
            if option_type == "call":
                ivs, new_strikes = skew_calc.calculate_call_skew(spot, mid_prices, strikes, dte/365, r = r, q = q)
                ivs, new_strikes = np.array(ivs), np.array(new_strikes)
                deltas = AnalyticalDelta().calculate(spot, new_strikes, dte/365, ivs, r, q, otype=option_type)

            elif option_type == "put":
                ivs, new_strikes = skew_calc.calculate_put_skew(spot, mid_prices, strikes, dte/365, r = r, q = q)
                ivs, new_strikes = np.array(ivs), np.array(new_strikes)
                deltas = AnalyticalDelta().calculate(spot, new_strikes, dte/365, ivs, r, q, otype=option_type)

            idx = np.argmin(np.abs(deltas - delta))
            data.append((key, float(mid_prices[idx]), float(new_strikes[idx]), dte, spot))

        return data

    def get_stock_price_history(self, ticker: str, start_date: str | None = None, end_date: str | None = None):
        query = """SELECT d.close_date, s.close AS spot
        FROM (
            SELECT DISTINCT close_date
            FROM option_data
            WHERE ticker = ?
        """
        params = [ticker]

        if start_date is not None:
            query += " AND close_date >= ?"
            params.append(start_date)

        if end_date is not None:
            query += " AND close_date <= ?"
            params.append(end_date)

        query += """
        ) d
        JOIN spot_prices s
        ON s.ticker = ?
        AND s.close_date = d.close_date
        ORDER BY d.close_date;
        """
        params.append(ticker)

        cur = self.conn.cursor()
        cur.execute(query, params)
        return cur.fetchall()
