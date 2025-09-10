import sqlite3
import pandas as pd
from backtester import run_backtest_for_ticker
from config import DatabaseConfig

def main():
    initial_capital = 100000
    brokerage_fee = 0.069  # Procentuell avgift

    # Definiera parametrarna som ska testas
    days_to_predict_list = [3, 5, 10, 15]
    threshold_list = [0.02, 0.03, 0.04]

    db_name = DatabaseConfig.DB_NAME
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        tickers_df = pd.read_sql_query("SELECT DISTINCT ticker FROM stocks_prepared", conn)
        tickers = tickers_df['ticker'].tolist()
        
        best_results = {}
        
        for ticker in tickers:
            print(f"\n\n===== Testar optimering för {ticker} =====")
            best_profit = -float('inf')
            best_params = {}
            
            for days in days_to_predict_list:
                for threshold in threshold_list:
                    print(f"\n--- Kör backtest för {ticker} med Dagar: {days}, Tröskel: {threshold*100:.0f}% ---")
                    
                    # Kör backtest och få tillbaka resultatet
                    profit, capital = run_backtest_for_ticker(
                        ticker,
                        initial_capital,
                        brokerage_fee,
                        days_to_predict=days,
                        threshold=threshold
                    )
                    
                    if profit > best_profit:
                        best_profit = profit
                        best_params = {'days_to_predict': days, 'threshold': threshold}
            
            # Lagra de bästa resultaten för varje ticker
            best_results[ticker] = {'best_profit': best_profit, 'best_params': best_params}
            
        print("\n\n===== Sammanfattning av de bästa resultaten =====")
        for ticker, result in best_results.items():
            print(f"\nTicker: {ticker}")
            print(f"Bästa parametrar: Dagar={result['best_params']['days_to_predict']}, Tröskel={result['best_params']['threshold']*100:.0f}%")
            print(f"Högsta vinst/förlust: {result['best_profit']:.2f} kr")
            print("--------------------------------------------------")

    except sqlite3.Error as e:
        print(f"Ett fel uppstod: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    main()