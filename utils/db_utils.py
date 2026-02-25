import sqlite3

from global_variables import OPTION_DB_DIR

# def get_connection(db_path=OPTION_DB_DIR):
#     return sqlite3.connect(db_path)


def get_connection(db_path):
    conn = sqlite3.connect(db_path)
    # speed/safety tweaks (optional, good defaults)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    # enlarge page cache (~200k * 4KB ≈ 800MB; adjust to taste)
    # conn.execute("PRAGMA cache_size=-200000;")
    return conn


def create_options_table(conn) -> None:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS option_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        close_date TEXT NOT NULL,          -- store as 'YYYY-MM-DD'
        expiration_date TEXT NOT NULL,
        dte INTEGER,
        option_type TEXT NOT NULL,         -- 'call' | 'put'
        strike REAL NOT NULL,
        price REAL,
        bid REAL,
        ask REAL,
        mid_price REAL,
        yahoo_iv REAL,
        volume REAL,
        open_interest REAL
    );

    -- Prevent duplicates (natural key)
    CREATE UNIQUE INDEX IF NOT EXISTS ux_option_unique
    ON option_data (ticker, close_date, expiration_date, option_type, strike);

    -- Cross-sectional lookups (close_date slice)
    CREATE INDEX IF NOT EXISTS idx_cs_matrix
    ON option_data (ticker, close_date, option_type, expiration_date, strike);

    -- Time-series lookups (over close_date for given strike/exp)
    CREATE INDEX IF NOT EXISTS idx_ts_matrix
    ON option_data (ticker, expiration_date, option_type, strike, close_date);
    """)
    conn.commit()