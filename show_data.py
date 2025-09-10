import sqlite3
import pandas as pd
from config import DatabaseConfig

def show_tables_and_data(db_name):
    """
    Visar de första raderna, kolumntyper och antalet rader
    för tabellerna i databasen.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        print("Ansluten till databasen.")
        cursor = conn.cursor()
        
        # Hämta en lista över alla tabeller
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for table_name in tables:
            table_name = table_name[0]
            print(f"\n--- Översikt för tabell: {table_name} ---")
            
            # Visa de första 5 raderna
            query = f"SELECT * FROM {table_name} LIMIT 5"
            df = pd.read_sql_query(query, conn)
            print("\nDe första 5 raderna:")
            print(df.head().to_string())
            
            # Visa kolumninformation
            print("\nKolumninformation:")
            df.info()
            
            # Räkna antalet rader
            row_count_query = f"SELECT COUNT(*) FROM {table_name}"
            row_count = pd.read_sql_query(row_count_query, conn).iloc[0, 0]
            print(f"\nAntal rader: {row_count}")

    except sqlite3.Error as e:
        print(f"Ett fel uppstod: {e}")
    finally:
        if conn:
            conn.close()
            print("Databasanslutning stängd.")

def main():
    db_name = DatabaseConfig.DB_NAME
    show_tables_and_data(db_name)

if __name__ == '__main__':
    main()