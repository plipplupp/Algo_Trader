import yfinance as yf
import sqlite3
import pandas as pd
from config import DataConfig, DatabaseConfig
from datetime import datetime, timedelta

def run_data_pipeline(tickers):
    """
    Samlar in, bearbetar och sparar historisk aktiedata i en SQLite-databas.
    Uppdaterar befintlig data om databasen redan existerar.
    """
    db_name = DatabaseConfig.DB_NAME
    conn = None
    
    print("Startar datainsamlingsflödet...")
    
    try:
        conn = sqlite3.connect(db_name)
        
        # Hämta en lista över tabeller i databasen
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stocks_prepared';")
        table_exists = cursor.fetchone() is not None
        
        for ticker in tickers:
            print(f"\n--- Bearbetar ticker: {ticker} ---")
            
            start_date = DataConfig.START_DATE
            
            # Kolla om tabellen finns och hämta det senaste datumet för den specifika tickern
            if table_exists:
                query = f"SELECT MAX(date) FROM stocks_prepared WHERE ticker = '{ticker}'"
                max_date_db = pd.read_sql_query(query, conn).iloc[0, 0]
                
                if max_date_db:
                    # Använder rätt formatsträng för att matcha datumet
                    start_date_obj = datetime.strptime(max_date_db, '%Y-%m-%d')
                    start_date_new = (start_date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
                    
                    # Kontrollera att startdatumet inte är efter dagens datum
                    if start_date_new > DataConfig.END_DATE:
                        print(f"Ingen ny data att hämta för {ticker} idag.")
                        print(f"--- Klart med {ticker} ---")
                        continue
                        
                    print(f"Databasen finns. Hämtar ny data från: {start_date_new}")
                    start_date = start_date_new

            # Hämta data från Yahoo Finance
            data = yf.download(ticker, start=start_date, end=DataConfig.END_DATE, interval="1d")
            
            if data.empty:
                print(f"Varning: Ingen ny data att spara för {ticker}.")
                print(f"--- Klart med {ticker} ---")
                continue
            
            # Kontrollera för MultiIndex och platta till kolumnnamnen
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]

            # Förbered data för lagring
            data.reset_index(inplace=True)
            data.rename(columns={'Date': 'date', 'Adj Close': 'adj_close'}, inplace=True)
            data['ticker'] = ticker
            
            # Spara data i SQLite
            data.to_sql('stocks_prepared', conn, if_exists='append', index=False)
            print(f"Sparade {len(data)} rader för {ticker} i databasen.")
            
            print(f"--- Klart med {ticker} ---")

        print("\nDatainsamlingsflödet är komplett.")

    except sqlite3.Error as e:
        print(f"Ett fel uppstod: {e}")
    finally:
        if conn:
            conn.close()
