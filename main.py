import pandas as pd
import json
from data_pipeline import run_data_pipeline
from portfolio_backtester import run_portfolio_backtest
from individual_optimizer import run_backtest_for_ticker
from config import DataConfig, BacktestConfig

def _print_best_report(report_dict):
    """En hjälpfunktion för att snyggt skriva ut den bästa klassifikationsrapporten."""
    if not report_dict:
        print("Ingen klassifikationsrapport kunde genereras.")
        return

    print("\nKlassifikationsrapport för BÄSTA modell:")
    print("-" * 45)
    print(f"{'Klass':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 45)
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            p = metrics.get('precision', 0)
            r = metrics.get('recall', 0)
            f1 = metrics.get('f1-score', 0)
            print(f"{label:<10} | {p:<10.2f} | {r:<10.2f} | {f1:<10.2f}")
    print("-" * 45)


def run_individual_optimization():
    """
    Orkestrerar optimeringen, visar bästa rapporten och sparar resultatet till en JSON-fil.
    """
    print("\n--- Startar optimering för enskilda aktier ---")
    
    tickers = DataConfig.TICKERS
    initial_capital = BacktestConfig.INITIAL_CAPITAL
    brokerage_fixed_fee = BacktestConfig.BROKERAGE_FIXED_FEE
    brokerage_percentage = BacktestConfig.BROKERAGE_PERCENTAGE
    
    days_options = [3, 5, 10, 15]
    threshold_options = [0.02, 0.03, 0.04]
    
    best_results_all_tickers = []
    best_params_overall = {}

    for ticker in tickers:
        best_profit = -float('inf')
        best_params_for_ticker = {}
        best_report_for_ticker = {}
        best_start_date = None
        best_end_date = None
        best_days_in_backtest = None

        print(f"\n{'='*10} Testar parametrar för {ticker.upper()} {'='*10}")
        for days in days_options:
            for threshold in threshold_options:
                print(f"Kör test: {days} dagar, {threshold*100:.0f}% tröskel...")
                
                profit, _, report, days_in_backtest, start_date, end_date = run_backtest_for_ticker(
                    ticker, initial_capital, brokerage_fixed_fee, 
                    brokerage_percentage, days, threshold
                )
                
                if profit > best_profit:
                    best_profit = profit
                    best_params_for_ticker = {'days': days, 'threshold': threshold}
                    best_report_for_ticker = report
                    best_start_date = start_date
                    best_end_date = end_date
                    best_days_in_backtest = days_in_backtest

        final_params = best_params_for_ticker
        best_params_overall[ticker] = final_params
        best_results_all_tickers.append({
            'ticker': ticker,
            'profit': best_profit,
            'params': final_params,
            'start_date': best_start_date,
            'end_date': best_end_date,
            'days_in_backtest': best_days_in_backtest,
            'report': best_report_for_ticker
        })
        
        # Snyggare utskrift
        profit_formatted = f"{int(best_profit):,}".replace(",", " ")
        # Ta bort klockslaget från datumen
        start_date_str = str(best_start_date).split(' ')[0] if best_start_date else 'N/A'
        end_date_str = str(best_end_date).split(' ')[0] if best_end_date else 'N/A'
        
        print(f"\n==> Bästa resultat för {ticker}: {final_params} (Vinst: {profit_formatted} kr under {best_days_in_backtest} handelsdagar, {start_date_str} till {end_date_str})")
        _print_best_report(best_report_for_ticker)

    print("\n--- Optimering slutförd! ---")
    
    # Sammanfattning
    print("\n" + "="*40)
    print(" SAMMANFATTNING AV BÄSTA RESULTAT ")
    print("="*40)
    for result in best_results_all_tickers:
        profit_formatted = f"{int(result['profit']):,}".replace(",", " ")
        start_date_str = str(result['start_date']).split(' ')[0] if result['start_date'] else 'N/A'
        end_date_str = str(result['end_date']).split(' ')[0] if result['end_date'] else 'N/A'
        
        print(f"Aktie: {result['ticker'].upper()} | Vinst: {profit_formatted} kr | Parametrar: {result['params']} | ({result['days_in_backtest']} handelsdagar, {start_date_str} till {end_date_str})")
    
    try:
        with open('optimal_params.json', 'w') as f:
            json.dump(best_params_overall, f, indent=4)
        print("\nDe optimala parametrarna har sparats till 'optimal_params.json'.")
    except Exception as e:
        print(f"Kunde inte spara parametrar till fil: {e}")


def main():
    """
    Huvudfunktionen som agerar som en meny för arbetsflödet.
    """
    while True:
        print("\n" + "="*40)
        print("VÄLJ EN ÅTGÄRD:")
        print("1. Uppdatera all aktiedata (Hämta & Beräkna)")
        print("2. Kör full portfölj-backtest")
        print("3. Hitta optimala parametrar för varje aktie")
        print("4. Avsluta")
        print("="*40)
        
        choice = input("Ditt val [1, 2, 3, 4]: ")
        
        if choice == '1':
            run_data_pipeline(DataConfig.TICKERS)
        
        elif choice == '2':
            print("\n--- Startar portfölj-backtest ---")
            run_portfolio_backtest(
                tickers=DataConfig.TICKERS,
                initial_capital=BacktestConfig.INITIAL_CAPITAL,
                brokerage_fixed_fee=BacktestConfig.BROKERAGE_FIXED_FEE,
                brokerage_percentage=BacktestConfig.BROKERAGE_PERCENTAGE
            )
        
        elif choice == '3':
            run_individual_optimization()
            
        elif choice == '4':
            print("Avslutar programmet.")
            break
            
        else:
            print("Ogiltigt val, försök igen.")

if __name__ == "__main__":
    main()
