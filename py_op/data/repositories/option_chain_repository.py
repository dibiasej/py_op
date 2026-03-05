

from py_op.global_variables import OPTION_DB_DIR
from py_op.utils.db_utils import get_connection

class OptionChainRepository:

    def __init__(self):
        self.conn = get_connection(OPTION_DB_DIR)

    def get_chain_snapshot(self, ticker: str, close_date: str, option_type: str = None, expiry: str = None, strike: float = None, strike_min: float = None, strike_max: float = None, dte_min: int = None, dte_max: int = None, dte: int = None) -> list[tuple]:
        query = """
        SELECT ticker, close_date, option_type, expiration_date, dte, strike, price, bid, ask, mid_price, yahoo_iv, volume, open_interest
        FROM option_data
        WHERE ticker = ? AND close_date = ?
        """
        params = [ticker.upper(), close_date]

        if option_type is not None:
            query += " AND option_type = ?"
            params.append(option_type)

        if expiry is not None:
            query += " AND expiration_date = ?"
            params.append(str(expiry))

        if strike_min is not None:
            query += " AND strike >= ?"
            params.append(float(strike_min))

        if strike_max is not None:
            query += " AND strike <= ?"
            params.append(float(strike_max))

        if strike is not None:
            query += " AND strike = ?"
            params.append(float(strike))

        if dte_min is not None:
            query += " AND dte >= ?"
            params.append(float(dte_min))

        if dte_max is not None:
            query += " AND dte <= ?"
            params.append(float(dte_max))

        if dte is not None:
            query += " AND dte = ?"
            params.append(float(dte))

        query += " ORDER BY expiration_date, option_type, strike"

        cur = self.conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()

        return rows
    
    def get_chain_timeseries(self, ticker: str, start_date: str, end_date: str, option_type: str = None, expiry: str = None, strike: float = None, strike_min: float = None, strike_max: float = None, dte_min: int = None, dte_max: int = None, dte: int = None, moneyness: float = None) -> list:

        query = """
        SELECT o.ticker, o.close_date, o.option_type, o.expiration_date,
        o.dte, o.strike, o.price, o.bid, o.ask, o.mid_price, 
        o.yahoo_iv, o.volume, o.open_interest,
        s.close AS spot
        FROM option_data o
        JOIN spot_prices s
            ON s.ticker = o.ticker
            AND s.close_date = o.close_date
        WHERE o.ticker = ?
        AND o.close_date BETWEEN ? AND ?
        """

        params: list = [ticker.upper(), start_date, end_date]

        if option_type is not None:
            query += " AND o.option_type = ?"
            params.append(option_type)

        if expiry is not None:
            query += " AND o.expiration_date = ?"
            params.append(str(expiry))

        if strike_min is not None:
            query += " AND o.strike >= ?"
            params.append(float(strike_min))

        if strike_max is not None:
            query += " AND o.strike <= ?"
            params.append(float(strike_max))

        if strike is not None:
            query += " AND o.strike = ?"
            params.append(float(strike))

        if dte_min is not None:
            query += " AND o.dte >= ?"
            params.append(float(dte_min))

        if dte_max is not None:
            query += " AND o.dte <= ?"
            params.append(float(dte_max))

        if dte is not None:
            query += " AND o.dte = ?"
            params.append(float(dte))

        if moneyness is not None:
            query += """
            AND s.close > 0
            AND ABS((CAST(o.strike AS REAL) / CAST(s.close AS REAL)) - 1.0) <= ?
            """
            params.append(float(moneyness))

        query += """
        ORDER BY o.close_date, o.expiration_date, o.option_type, o.strike
        """

        cur = self.conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()

        return rows

    
    def get_common_expirations(self, ticker, close_date):
        # This query fetches common expirations in a chain for puts and calls for a specific ticker on a specific date
        query = """
        SELECT expiration_date, dte
        FROM option_data
        WHERE ticker = ? AND close_date = ? AND option_type = 'call'
        INTERSECT
        SELECT expiration_date, dte
        FROM option_data
        WHERE ticker = ? AND close_date = ? AND option_type = 'put'
        ORDER BY expiration_date
        """

        params=[ticker, close_date, ticker, close_date]
        cur = self.conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()
        return rows