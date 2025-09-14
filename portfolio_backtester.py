import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
import math
from config import DatabaseConfig
from utils import calculate_brokerage_fee, create_future_label, calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown

def _calculate_portfolio_buy_and_hold(prepared_dfs, initial_capital, tickers, start_date, end_date):
    """
    Beräknar avkastningen för en "köp och behåll"-strategi för hela portföljen.
    """
    capital_per_ticker = initial_capital / len(tickers)
    total_final_value = 0
    
    for ticker in tickers:
        df = prepared_dfs[ticker]
        
        # Hitta start- och slutpris så nära periodens datum som möjligt
        start_price_row = df[df['date'] >= start_date].iloc[0]
        end_price_row = df[df['date'] <= end_date].iloc[-1]
        
        start_price = start_price_row['adj_close']
        end_price = end_price_row['adj_close']
        
        if start_price > 0:
            shares_bought = capital_per_ticker / start_price
            total_final_value += shares_bought * end_price
            
    return total_final_value - initial_capital

def run_portfolio_backtest(tickers, initial_capital, brokerage_fixed_fee, brokerage_percentage):
    print("\n--- Startar Walk-Forward Portfölj-backtest ---")
    db_name = DatabaseConfig.DB_NAME

    try:
        with open('optimal_params.json', 'r') as f:
            optimal_params = json.load(f)
    except FileNotFoundError:
        print("Kunde inte hitta 'optimal_params.json'. Kör alternativ 3 (optimering) först.")
        return

    prepared_dfs = {}
    try:
        conn = sqlite3.connect(db_name)
        for ticker in tickers:
            df = pd.read_sql_query(f"SELECT * FROM stocks_prepared WHERE ticker = '{ticker}'", conn)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                prepared_dfs[ticker] = df.sort_values('date').reset_index(drop=True)
    finally:
        if conn:
            conn.close()

    if not prepared_dfs:
        print("Ingen data hittades för angivna tickers.")
        return

    master_df = pd.concat(prepared_dfs.values()).sort_values('date').reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=4)
    
    current_capital = initial_capital
    positions = {ticker: {'shares': 0, 'purchase_price': 0} for ticker in tickers}
    daily_portfolio_history = []
    
    # Räknare för avgifter och transaktioner
    total_brokerage_fees = 0.0
    total_transactions = 0
    
    # Hela backtest-periodens start och slut
    backtest_indices = [i for train_index, test_index in tscv.split(master_df) for i in test_index]
    if not backtest_indices:
        print("Inga testperioder genererades. Kontrollera din data och TimeSeriesSplit-inställningar.")
        return

    backtest_start_date = master_df.iloc[backtest_indices[0]]['date']
    backtest_end_date = master_df.iloc[backtest_indices[-1]]['date']
    
    for train_index, test_index in tscv.split(master_df):
        train_df = master_df.iloc[train_index]
        test_df = master_df.iloc[test_index]
        
        print(f"\nProcessing fold: Train [{train_df['date'].min().date()}:{train_df['date'].max().date()}], Test [{test_df['date'].min().date()}:{test_df['date'].max().date()}]")

        models = {}
        for ticker in tickers:
            ticker_train_df = train_df[train_df['ticker'] == ticker].copy()
            params = optimal_params.get(ticker, {'days': 10, 'threshold': 0.03})
            
            ticker_train_df = create_future_label(ticker_train_df, days=params['days'], threshold=params['threshold'])
            ticker_train_df.dropna(subset=['future_label'], inplace=True)

            if len(ticker_train_df) < 50: continue

            train_features = ticker_train_df.drop(columns=['ticker', 'future_label', 'date'])
            train_labels = ticker_train_df['future_label']
            
            le = LabelEncoder()
            train_labels_encoded = le.fit_transform(train_labels)
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'))
            ])
            pipeline.fit(train_features, train_labels_encoded)
            models[ticker] = (pipeline, le)

        # Simulera handel dag för dag genom testperioden för denna fold
        for date in sorted(test_df['date'].unique()):
            date = pd.to_datetime(date)
            # Säljlogik
            for ticker, pos in list(positions.items()):
                if pos['shares'] > 0:
                    day_data = test_df[(test_df['date'] == date) & (test_df['ticker'] == ticker)]
                    if day_data.empty: continue

                    current_price = day_data['adj_close'].iloc[0]
                    prediction = 'Behåll' # Default
                    if ticker in models:
                        pipeline, le = models[ticker]
                        features = day_data.drop(columns=['ticker', 'date'])
                        train_cols = pipeline.named_steps['classifier'].get_booster().feature_names
                        features = features.reindex(columns=train_cols, fill_value=0)
                        prediction_encoded = pipeline.predict(features)
                        prediction = le.inverse_transform(prediction_encoded)[0]

                    if prediction == 'Sälj' or current_price <= pos['purchase_price'] * 0.95:
                        sale_cost = pos['shares'] * current_price
                        sale_fee = calculate_brokerage_fee(sale_cost, brokerage_fixed_fee, brokerage_percentage)
                        current_capital += sale_cost - sale_fee
                        total_brokerage_fees += sale_fee
                        total_transactions += 1
                        print(f"  - {str(date.date())}: SÄLJ {pos['shares']} st {ticker} @ {current_price:.2f}. Kapital: {current_capital:,.0f} kr")
                        positions[ticker] = {'shares': 0, 'purchase_price': 0}

            # Köplogik
            buy_signals_today = []
            for ticker in tickers:
                if ticker in models:
                    day_data = test_df[(test_df['date'] == date) & (test_df['ticker'] == ticker)]
                    if day_data.empty: continue
                    pipeline, le = models[ticker]
                    features = day_data.drop(columns=['ticker', 'date'])
                    train_cols = pipeline.named_steps['classifier'].get_booster().feature_names
                    features = features.reindex(columns=train_cols, fill_value=0)
                    prediction_encoded = pipeline.predict(features)
                    prediction = le.inverse_transform(prediction_encoded)[0]
                    if prediction == 'Köp':
                        buy_signals_today.append({'ticker': ticker, 'price': day_data['adj_close'].iloc[0]})

            if buy_signals_today and current_capital > brokerage_fixed_fee:
                capital_per_trade = current_capital / len(buy_signals_today)
                for signal in buy_signals_today:
                    ticker, price = signal['ticker'], signal['price']
                    if price > 0 and positions[ticker]['shares'] == 0:
                        shares_to_buy = math.floor(capital_per_trade / price)
                        while shares_to_buy > 0:
                            buy_cost = (shares_to_buy * price) + calculate_brokerage_fee(shares_to_buy * price, brokerage_fixed_fee, brokerage_percentage)
                            if buy_cost <= capital_per_trade: break
                            shares_to_buy -= 1
                        if shares_to_buy > 0:
                            buy_cost = (shares_to_buy * price) + calculate_brokerage_fee(shares_to_buy * price, brokerage_fixed_fee, brokerage_percentage)
                            positions[ticker] = {'shares': shares_to_buy, 'purchase_price': price}
                            current_capital -= buy_cost
                            total_brokerage_fees += calculate_brokerage_fee(shares_to_buy * price, brokerage_fixed_fee, brokerage_percentage)
                            total_transactions += 1
                            print(f"  - {str(date.date())}: KÖP {shares_to_buy} st {ticker} @ {price:.2f}. Kapital: {current_capital:,.0f} kr")

            # Beräkna portföljens värde vid dagens slut
            portfolio_value = current_capital
            for ticker, pos in positions.items():
                if pos['shares'] > 0:
                    last_price_data = prepared_dfs[ticker][prepared_dfs[ticker]['date'] <= date]
                    if not last_price_data.empty:
                        portfolio_value += pos['shares'] * last_price_data['adj_close'].iloc[-1]
            daily_portfolio_history.append({'date': date, 'portfolio_value': portfolio_value})

    # --- SLUTSUMMERING (omskriven för att inkludera all info) ---
    if not daily_portfolio_history:
        print("Inga affärer gjordes under backtest-perioden.")
        return
        
    final_portfolio_value = daily_portfolio_history[-1]['portfolio_value']
    total_profit = final_portfolio_value - initial_capital
    
    history_df = pd.DataFrame(daily_portfolio_history).set_index('date')
    daily_returns = history_df['portfolio_value'].pct_change().dropna()
    
    sharpe = calculate_sharpe_ratio(daily_returns)
    sortino = calculate_sortino_ratio(daily_returns)
    max_dd = calculate_max_drawdown(history_df['portfolio_value'])

    # Beräkna "Köp och Behåll"
    buy_and_hold_profit = _calculate_portfolio_buy_and_hold(prepared_dfs, initial_capital, tickers, backtest_start_date, backtest_end_date)
    buy_and_hold_capital = initial_capital + buy_and_hold_profit

    print("\n" + "--- Slutresultat ---")
    print(f"Initialt kapital: {initial_capital:.0f} kr")
    print(f"Slutkapital med portföljstrategi: {final_portfolio_value:.0f} kr")
    print(f"Total vinst/förlust: {total_profit:.0f} kr ({(total_profit/initial_capital)*100:.2f}%)")
    print(f"Total courtageavgift: {total_brokerage_fees:.0f} kr")
    print(f"Totalt antal transaktioner: {total_transactions}")
    print("-" * 30)
    print(f"Sharpekvot (årlig): {sharpe:.2f}")
    print(f"Sortinokvot (årlig): {sortino:.2f}")
    print(f"Maximal nedgång (Drawdown): {max_dd:.2%}")
    print("\n" + "--- Jämförelse med Portfölj Köp & Behåll ---")
    print(f"Slutkapital med 'Köp och Behåll': {buy_and_hold_capital:.0f} kr")
    print(f"Total vinst/förlust: {buy_and_hold_profit:.0f} kr ({(buy_and_hold_profit/initial_capital)*100:.2f}%)")
