import yfinance as yf
import sqlite3
import pandas as pd
from config import DataConfig, DatabaseConfig
from datetime import datetime, timedelta
from utils import calculate_atr

def _create_tables(conn):
    """Skapar databastabellerna om de inte finns."""
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stocks_raw (
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL, adj_close REAL, volume INTEGER,
            PRIMARY KEY (date, ticker)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stocks_prepared (
            date TEXT, ticker TEXT, open REAL, high REAL, low REAL, close REAL, 
            adj_close REAL, volume REAL, sma_10 REAL, sma_50 REAL, sma_100 REAL, 
            sma_200 REAL, ema_10 REAL, ema_50 REAL, ema_100 REAL, ema_200 REAL, 
            rsi_14 REAL, macd REAL, macd_signal REAL, bollinger_middle REAL, 
            bollinger_upper REAL, bollinger_lower REAL, daily_range REAL, 
            volume_sma_20 REAL, atr_14 REAL, vix_close REAL,
            PRIMARY KEY (date, ticker)
        )
    ''')
    conn.commit()

def _fetch_raw_data(tickers, conn):
    """Steg 1: Hämtar rådata inkrementellt för aktier och VIX."""
    print("--- Startar Steg 1: Hämtar rådata ---")
    
    symbols_to_fetch = tickers + ['^VIX']

    for symbol in symbols_to_fetch:
        db_ticker_name = 'VIX' if symbol == '^VIX' else symbol
        print(f"Hämtar data för {db_ticker_name}...")
        
        start_date = DataConfig.START_DATE
        try:
            query = f"SELECT MAX(date) FROM stocks_raw WHERE ticker = '{db_ticker_name}'"
            max_date_db = pd.read_sql(query, conn).iloc[0, 0]
            if max_date_db:
                start_date_obj = datetime.strptime(max_date_db.split(' ')[0], '%Y-%m-%d')
                start_date = (start_date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
        except (pd.io.sql.DatabaseError, IndexError, TypeError):
            pass

        if start_date >= DataConfig.END_DATE:
            print(f"Data för {db_ticker_name} är redan uppdaterad.")
            continue
            
        ### FÖRBÄTTRING: auto_adjust=False säkerställer att 'Adj Close' finns med.
        data = yf.download(symbol, start=start_date, end=DataConfig.END_DATE, auto_adjust=False)
        if data.empty:
            print(f"Ingen ny data hittades för {db_ticker_name}.")
            continue
            
        data.reset_index(inplace=True)
        
        ### FÖRBÄTTRING: En mer robust metod för att standardisera kolumnnamn.
        data.columns = [str(col).lower().replace(' ', '_') for col in data.columns]
        
        data['date'] = data['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        data['ticker'] = db_ticker_name
        
        db_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        data_to_save = data[[col for col in db_cols if col in data.columns]]
        
        data_to_save.to_sql('stocks_raw', conn, if_exists='append', index=False)
        print(f"Sparade {len(data_to_save)} nya rader rådata för {db_ticker_name}.")

def _calculate_and_save_features(tickers, conn):
    """Steg 2: Beräknar features, hanterar helgdagar, slår ihop med VIX och sparar."""
    print("\n--- Startar Steg 2: Beräknar tekniska indikatorer ---")
    
    cursor = conn.cursor()
    cursor.execute("DELETE FROM stocks_prepared;")
    conn.commit()
    print("Rensat 'stocks_prepared' för att bygga upp den på nytt.")

    vix_df = pd.read_sql("SELECT date, adj_close as vix_close FROM stocks_raw WHERE ticker='VIX' ORDER BY date", conn)
    if not vix_df.empty:
        vix_df['date'] = pd.to_datetime(vix_df['date']).dt.date.astype(str)

    for ticker in tickers:
        print(f"Beräknar features för {ticker}...")
        df = pd.read_sql(f"SELECT * FROM stocks_raw WHERE ticker='{ticker}' ORDER BY date", conn)
        if df.empty:
            continue
        
        required_cols = ['adj_close', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            print(f"Varning: Nödvändiga kolumner saknas för {ticker}. Skippar.")
            continue

        ### FÖRBÄTTRING: Fyller igen helgdagar med föregående dags data (forward-fill)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        df = df.reindex(full_date_range)
        df.fillna(method='ffill', inplace=True)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'date'}, inplace=True)
        df['ticker'] = ticker # Fyll i ticker-namnet som försvann vid reindex

        df['sma_10'] = df['adj_close'].rolling(window=10).mean()
        df['sma_50'] = df['adj_close'].rolling(window=50).mean()
        df['sma_100'] = df['adj_close'].rolling(window=100).mean()
        df['sma_200'] = df['adj_close'].rolling(window=200).mean()
        df['ema_10'] = df['adj_close'].ewm(span=10, adjust=False).mean()
        df['ema_50'] = df['adj_close'].ewm(span=50, adjust=False).mean()
        df['ema_100'] = df['adj_close'].ewm(span=100, adjust=False).mean()
        df['ema_200'] = df['adj_close'].ewm(span=200, adjust=False).mean()
        delta = df['adj_close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain.div(loss)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        ema_12 = df['adj_close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['adj_close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        bollinger_window = 20
        df['bollinger_middle'] = df['adj_close'].rolling(window=bollinger_window).mean()
        std_dev = df['adj_close'].rolling(window=bollinger_window).std()
        df['bollinger_upper'] = df['bollinger_middle'] + 2 * std_dev
        df['bollinger_lower'] = df['bollinger_middle'] - 2 * std_dev
        df['daily_range'] = df['high'] - df['low']
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['atr_14'] = calculate_atr(df['high'], df['low'], df['close'], window=14)
        
        if not vix_df.empty:
            df['date_only'] = pd.to_datetime(df['date']).dt.date.astype(str)
            df = pd.merge(df, vix_df, left_on='date_only', right_on='date', how='left', suffixes=('', '_vix'))
            df.drop(columns=['date_only', 'date_vix'], inplace=True, errors='ignore')
            df['vix_close'] = df['vix_close'].fillna(method='ffill')
        else:
            df['vix_close'] = 0

        df.to_sql('stocks_prepared', conn, if_exists='append', index=False)
        print(f"Sparade {len(df)} rader med features för {ticker}.")

def run_data_pipeline(tickers):
    """Huvudfunktion som orkestrerar hela datainsamlings- och bearbetningsflödet."""
    db_name = DatabaseConfig.DB_NAME
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        _create_tables(conn)
        _fetch_raw_data(tickers, conn)
        _calculate_and_save_features(tickers, conn)
        print("\nDatapipelinen är färdig!")
    except Exception as e:
        print(f"Ett fel uppstod i datainsamlingen: {e}")
    finally:
        if conn:
            conn.close()