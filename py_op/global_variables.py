import datetime as dt

TICKERS: list[str] = ["SPY", "QQQ", "TLT", "IEF", "SHY", "HYG", "LQD", "GLD", "SLV", "UCO", "^vix", "vt", 'XLE',
               	'XLF',	'XLU',	'XLI',	'GDX',	'XLK',	'XLV',	'XLY',	'XLP',	'XLB',	'XOP',	'IYR',	'XHB',
                'ITB',	'VNQ',	'GDXJ',	'IYE',	'OIH',	'XME',	'XRT',	'SMH',	'IBB', "VXX", "UVXY", "SVIX", 'KBE', 'KRE',	'XTL', 'IWM',
                'IJR', 'IWO', 'SPMD', 'SPEU', 'SPEM', 'SPSM', 'SPYG', 'SPYV', 'MDYG', 'MDYV', 'SLYG', 'SLYV', 'KIE',
                'XAR', 'XTN', 'XBI', 'XPH', 'XHS', 'XHE', 'XSW', 'XSD', 'TSLA', 'NVDA', 'AMZN', 'AAPL', 'AMD', 
                'BABA', 'PLTR', 'MSFT', 'INTC', 'BAC', 'META', 'GOOGL', "MARA", 'BA', 'PFE', 'NIO', 'ORCL', 'UBER', 'RIVN', 
                'SOFI', 'AVGO', 'COST', 'NFLX', 'MRNA', 'ADBE', 'COIN', 'ENPH', 'ZM', 'TQQQ', 'CVNA', 'PYPL', 'CRWD', 'LLY',
                'JPM', 'EEM', 'CRM', 'USB', 'XOM', 'DIS', 'HD', 'GS', 'RTX', 'JD', 'CVX', 'WFC', 'C', 'JNJ', 'KO', 'WMT',
                'AFRM', 'ROKU', 'PEP', 'U', 'TGT', 'SNOW', 'SHOP', 'CMG', 'LULU', 'LMT', 'MS', 'NOW', 'LYFT', 'NKE', 'CAT',
                'UNH', 'ASML', 'GM', 'TSM', 'KOLD', 'AXP', 'EWZ', 'SCHW', 'SBUX', 'MCD', 'V', 'FDX', 'PINS', 'GE', 'DAL',
                'ABBV', 'MA', 'F', 'VZ', 'ULTA', 'Z', 'RCL', 'SPOT', 'CVS', 'AAL', 'CCL', 'DASH', 'T', 'BLK', 'DPZ', 'EBAY',
                'TJX', '^NDX', '^SPX', '^RUT', 'TIP', 'MBB', "HON", "UPS", "DE", "CME", "ICE", "AMGN", "HUM", "GILD", "COP",
                "SLB", "HAL", "LVS", "MGM", "RIOT", "MSTR", "AIG", "PG", "LOW", "CSCO", "QCOM", "MU", "AMAT", "UUP", "UDN", "FXE",
                "FXB", "FXY", "JNK", "AGG", "UNG", "USO", "PPLT", "PALL", "CORN", "SOYB", "WEAT"]

TODAY: str = dt.datetime.now().strftime('%Y-%m-%d')

DIR: str = "/Users/dibia/OneDrive/Documents/Data/Options/options chain/"

OPTION_DB_DIR = r"C:\Users\dibia\OneDrive\Documents\Data\Options\options chain\option_data.db"