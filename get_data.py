import sqlite3
import pandas as pd
import yfinance as yf
import os

from config import DatabaseConfig, DataConfig

def create_db_table(db_name, table_name='stocks'):
    """
    Skapar databastabellen med alla nödvändiga kolumner.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adj_close REAL,
                volume INTEGER,
                PRIMARY KEY (date, ticker)
            )
        ''')
        conn.commit()
    except sqlite3.Error as e:
        print(f"Ett fel uppstod vid skapandet av tabell: {e}")
    finally:
        if conn:
            conn.close()

def get_raw_data(ticker, period="5y"):
    """
    Hämtar historisk rådata från yfinance för en given ticker och tidsperiod.
    """
    try:
        data = yf.download(ticker, period=period, progress=False, auto_adjust=False)
        
        if data.empty:
            print(f"Ingen data kunde hämtas för {ticker}.")
            return pd.DataFrame()
            
        # Hantera MultiIndex-problemet genom att rensa kolumnerna
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        
        data.index = pd.to_datetime(data.index).strftime('%Y-%m-%d')
        data.index.name = 'date'
        data['ticker'] = ticker
        
        # Konvertera relevanta kolumner till gemener
        data.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        }, inplace=True)
        
        return data
        
    except Exception as e:
        print(f"Ett fel uppstod vid hämtning av data för {ticker}: {e}")
        return pd.DataFrame()

def save_to_db(df, db_name, table_name='stocks'):
    """
    Sparar DataFrame till en SQLite-databas.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        
        # Kontrollera och rensa gamla rader för den aktuella tickern
        ticker = df['ticker'].iloc[0]
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {table_name} WHERE ticker = ?", (ticker,))
        conn.commit()

        # Spara data till databasen
        df.to_sql(table_name, conn, if_exists='append', index=True)
        print(f"Sparade {len(df)} rader för {ticker} i databasen.")
        
    except sqlite3.Error as e:
        print(f"Ett fel uppstod vid sparande till databas: {e}")
    finally:
        if conn:
            conn.close()

def run_data_pipeline():
    """
    Huvudfunktionen som orkestrerar hela datainsamlings- och bearbetningsflödet.
    """
    db_name = DatabaseConfig.DB_NAME
    create_db_table(db_name)
    
    tickers = DataConfig.TICKERS
    
    print("Startar datainsamlingsflödet...")
    
    for ticker in tickers:
        print(f"\n--- Bearbetar ticker: {ticker} ---")
        
        raw_data = get_raw_data(ticker)
        
        if not raw_data.empty:
            save_to_db(raw_data, db_name)
        
        print(f"--- Klart med {ticker} ---")
    
    print("\nDatainsamlingsflödet är komplett.")

if __name__ == '__main__':
    run_data_pipeline()