import sqlite3
from typing import Dict, Any, List, Tuple

from utils.db_utils import get_connection
from data.price_data.load_price_data import load_all_price_data
from global_variables import OPTION_DB_DIR

def ensure_spot_prices_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS spot_prices (
        ticker      TEXT NOT NULL,
        close_date  TEXT NOT NULL,
        open        REAL,
        high        REAL,
        low         REAL,
        close       REAL NOT NULL,
        volume      REAL,
        PRIMARY KEY (ticker, close_date)
    );
    """)
    conn.execute("""
    CREATE INDEX IF NOT EXISTS idx_spot_prices_close_date
    ON spot_prices(close_date);
    """)
    conn.commit()

def upsert_spot_prices(conn: sqlite3.Connection, all_price_dict: Dict[str, Dict[str, Any]]) -> int:
    """
    all_price_dict[ticker] = {"Date": np.array([...]), "Open":..., "Close":..., ...}
    """
    rows: List[Tuple] = []

    for ticker, d in all_price_dict.items():
        dates = d["Date"]
        opens = d.get("Open")
        highs = d.get("High")
        lows  = d.get("Low")
        closes = d["Close"]
        vols = d.get("Volume")

        n = len(dates)
        for i in range(n):
            rows.append((
                str(ticker).upper(),
                str(dates[i]),
                float(opens[i]) if opens is not None else None,
                float(highs[i]) if highs is not None else None,
                float(lows[i])  if lows  is not None else None,
                float(closes[i]),
                float(vols[i]) if vols is not None else None,
            ))

    # UPSERT so you can rerun safely
    conn.executemany("""
    INSERT INTO spot_prices (ticker, close_date, open, high, low, close, volume)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(ticker, close_date) DO UPDATE SET
        open=excluded.open,
        high=excluded.high,
        low=excluded.low,
        close=excluded.close,
        volume=excluded.volume;
    """, rows)
    conn.commit()
    return len(rows)

def build_spot_table_main():
    conn = get_connection(OPTION_DB_DIR)
    ensure_spot_prices_table(conn)

    all_price_dict = load_all_price_data()  # from your load_price_data.py
    inserted = upsert_spot_prices(conn, all_price_dict)
    print(f"Upserted {inserted:,} spot rows.")
    conn.close()