import sqlite3
import pandas as pd
from config import DatabaseConfig, DataConfig

def calculate_technical_indicators(df):
    """
    Beräknar tekniska indikatorer och lägger till dem i DataFrame.
    """
    if df.empty:
        return pd.DataFrame()
    
    required_cols = ['adj_close', 'high', 'low', 'volume']
    if not all(col in df.columns for col in required_cols):
        print("Saknar en eller flera nödvändiga kolumner.")
        return pd.DataFrame()

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
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    ema_12 = df['adj_close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['adj_close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    bollinger_window = 20
    df['bollinger_middle'] = df['adj_close'].rolling(window=bollinger_window).mean()
    df['bollinger_upper'] = df['bollinger_middle'] + 2 * df['adj_close'].rolling(window=bollinger_window).std()
    df['bollinger_lower'] = df['bollinger_middle'] - 2 * df['adj_close'].rolling(window=bollinger_window).std()

    df['daily_range'] = df['high'] - df['low']
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    
    return df

def create_future_label(df, days=10, threshold=0.03):
    """
    Skapar målvariabeln (label) baserat på framtida prisrörelse.
    """
    if df.empty or 'adj_close' not in df.columns:
        return df
        
    price_future = df['adj_close'].shift(-days)
    price_change = (price_future - df['adj_close']) / df['adj_close']
    
    df['future_label'] = 'Behåll'
    df.loc[price_change >= threshold, 'future_label'] = 'Köp'
    df.loc[price_change <= -threshold, 'future_label'] = 'Sälj'
    
    df.loc[df.index.isin(df.index[-days:]), 'future_label'] = None
    
    return df

def run_full_pipeline():
    """
    Huvudfunktionen som orkestrerar hela flödet för features och labels.
    """
    db_name = DatabaseConfig.DB_NAME
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        
        # Hämta alla unika tickrar från databasen
        tickers_df = pd.read_sql_query("SELECT DISTINCT ticker FROM stocks", conn)
        tickers = tickers_df['ticker'].tolist()
        
        # Skapa en ny tabell för beräknade features.
        # Detta garanterar en ren start varje gång.
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS stocks_prepared")
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS stocks_prepared (
                date TEXT,
                ticker TEXT,
                open REAL, high REAL, low REAL, close REAL, adj_close REAL, volume REAL,
                sma_10 REAL, sma_50 REAL, sma_100 REAL, sma_200 REAL,
                ema_10 REAL, ema_50 REAL, ema_100 REAL, ema_200 REAL,
                rsi_14 REAL, macd REAL, macd_signal REAL,
                bollinger_middle REAL, bollinger_upper REAL, bollinger_lower REAL,
                daily_range REAL, volume_sma_20 REAL,
                future_label TEXT,
                PRIMARY KEY (date, ticker)
            )
        ''')
        conn.commit()
        
        for ticker in tickers:
            print(f"\n--- Beräknar features och labels för {ticker} ---")
            
            query = f"SELECT * FROM stocks WHERE ticker = '{ticker}' ORDER BY date"
            df = pd.read_sql_query(query, conn)
            
            if df.empty:
                print(f"Ingen data att beräkna för {ticker}.")
                continue
            
            df = df.set_index('date')
            for col in ['open', 'high', 'low', 'close', 'adj_close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df_with_features = calculate_technical_indicators(df)
            df_with_features = create_future_label(df_with_features)
            
            if not df_with_features.empty:
                df_with_features.to_sql('stocks_prepared', conn, if_exists='append', index=True)
                print(f"Sparade komplett data för {ticker} i databasen.")
            
    except sqlite3.Error as e:
        print(f"Ett fel uppstod: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    run_full_pipeline()