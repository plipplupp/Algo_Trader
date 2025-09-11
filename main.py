import pandas as pd
from data_collector import run_data_pipeline
from portfolio_backtester import run_portfolio_backtest
from config import DataConfig, BacktestConfig

def main():
    """
    Huvudfunktionen som orkestrerar hela arbetsflödet.
    """
    tickers = DataConfig.TICKERS
    initial_capital = BacktestConfig.INITIAL_CAPITAL
    
    # Hämta de nya variablerna för courtageavgiften från config.py
    brokerage_fixed_fee = BacktestConfig.BROKERAGE_FIXED_FEE
    brokerage_percentage = BacktestConfig.BROKERAGE_PERCENTAGE

    # 1. Samla in och förbered data för alla tickers
    run_data_pipeline(tickers)

    # 2. Definiera de optimala parametrarna manuellt
    # Dessa är baserade på din tidigare analys.
    optimal_params = {
        'AAPL': {'days': 5, 'threshold': 0.03},
        'GOOGL': {'days': 5, 'threshold': 0.03},
        'MSFT': {'days': 5, 'threshold': 0.03},
        'NVDA': {'days': 10, 'threshold': 0.04}, # Använder det bästa resultatet med stop-loss
        'SWED-A.ST': {'days': 15, 'threshold': 0.04},
        'TSLA': {'days': 10, 'threshold': 0.02}
    }
    
    # 3. Kör den nya portfölj-backtestern och skicka med de nya avgiftsvärdena
    run_portfolio_backtest(tickers, optimal_params, initial_capital, brokerage_fixed_fee, brokerage_percentage)

if __name__ == "__main__":
    main()
