
from py_op.global_variables import OPTION_DB_DIR
from py_op.utils.db_utils import get_connection

class PositionSeriesRepository:

    def __init__(self):
        self.conn = get_connection(OPTION_DB_DIR)

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
