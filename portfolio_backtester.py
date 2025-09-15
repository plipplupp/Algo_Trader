import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
import math
from config import DatabaseConfig
from utils import calculate_brokerage_fee, create_future_label, calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown

# =============================================================================
# NY STRATEGI-INSTÄLLNING
# =============================================================================
# Styr hur stor andel av TILLGÄNGLIGT KAPITAL som ska användas för en dags köp.
# 1.0 = Använd allt tillgängligt kapital.
# 0.5 = Använd hälften av tillgängligt kapital, spara resten.
# 0.25 = Använd en fjärdedel.
TRADE_CAPITAL_ALLOCATION = 0.8
# =============================================================================

def select_best_features(features_df, labels, top_n=60):
    model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
    sample_weights = compute_sample_weight(class_weight='balanced', y=labels)
    model.fit(features_df, labels, sample_weight=sample_weights)
    
    feature_importances = pd.Series(model.feature_importances_, index=features_df.columns)
    best_features = feature_importances.nlargest(top_n).index.tolist()
    return best_features

def _calculate_portfolio_buy_and_hold(prepared_dfs, initial_capital, tickers, start_date, end_date):
    capital_per_ticker = initial_capital / len(tickers) if tickers else 0
    total_final_value = 0
    
    for ticker in tickers:
        df = prepared_dfs.get(ticker)
        if df is None: continue
        
        start_price_rows = df[df['date'] >= start_date]
        if start_price_rows.empty: continue
        start_price = start_price_rows.iloc[0]['adj_close']
        
        end_price_rows = df[df['date'] <= end_date]
        if end_price_rows.empty: continue
        end_price = end_price_rows.iloc[-1]['adj_close']
        
        if start_price > 0:
            shares_bought = capital_per_ticker / start_price
            total_final_value += shares_bought * end_price
            
    return total_final_value

def run_portfolio_backtest(tickers, initial_capital, brokerage_fixed_fee, brokerage_percentage):
    print(f"\n--- Startar Walk-Forward Portfölj-backtest (Kapitalallokering per dag: {TRADE_CAPITAL_ALLOCATION:.0%}) ---")
    db_name = DatabaseConfig.DB_NAME

    try:
        with open('optimal_params.json', 'r') as f:
            optimal_params = json.load(f)
    except FileNotFoundError:
        print("Kunde inte hitta 'optimal_params.json'. Kör alternativ 3 (optimering) först.")
        return

    prepared_dfs = {}
    conn = sqlite3.connect(db_name)
    try:
        for ticker in tickers:
            df = pd.read_sql_query(f"SELECT * FROM stocks_prepared WHERE ticker = '{ticker}'", conn)
            df = df.loc[:,~df.columns.duplicated()]
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                prepared_dfs[ticker] = df.sort_values('date').reset_index(drop=True)
    finally:
        conn.close()

    if not prepared_dfs:
        print("Ingen data hittades för angivna tickers.")
        return

    master_df = pd.concat(prepared_dfs.values()).sort_values('date').reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=4)
    
    current_capital = initial_capital
    positions = {ticker: {'shares': 0, 'purchase_price': 0} for ticker in tickers}
    daily_portfolio_history = []
    total_brokerage_fees = 0.0
    total_transactions = 0
    
    backtest_indices = [i for _, test_index in tscv.split(master_df) for i in test_index]
    if not backtest_indices:
        print("Inga testperioder genererades.")
        return
    backtest_start_date = master_df.iloc[min(backtest_indices)]['date']
    backtest_end_date = master_df.iloc[max(backtest_indices)]['date']
    
    for train_index, test_index in tscv.split(master_df):
        train_df = master_df.iloc[train_index]
        test_df = master_df.iloc[test_index]
        
        print(f"\nProcessing fold: Train [{train_df['date'].min().date()}:{train_df['date'].max().date()}], Test [{test_df['date'].min().date()}:{test_df['date'].max().date()}]")

        models = {}
        for ticker in tickers:
            # (Träningslogiken är oförändrad från din version, den är korrekt)
            ticker_train_df = train_df[train_df['ticker'] == ticker].copy()
            params = optimal_params.get(ticker, {'days': 10, 'threshold': 0.03})
            ticker_train_df = create_future_label(ticker_train_df, days=params['days'], threshold=params['threshold'])
            ticker_train_df.dropna(subset=['future_label'], inplace=True)
            if len(ticker_train_df) < 100: continue
            train_features_all = ticker_train_df.drop(columns=['ticker', 'future_label', 'date'])
            train_labels = ticker_train_df['future_label']
            le = LabelEncoder()
            train_labels_encoded = le.fit_transform(train_labels)
            best_features = select_best_features(train_features_all, train_labels_encoded)
            train_features_selected = train_features_all[best_features]
            pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'))])
            sample_weights = compute_sample_weight(class_weight='balanced', y=train_labels_encoded)
            pipeline.fit(train_features_selected, train_labels_encoded, classifier__sample_weight=sample_weights)
            models[ticker] = (pipeline, le, best_features)

        for date in sorted(test_df['date'].unique()):
            date = pd.to_datetime(date)
            
            # Säljlogik (från föregående svar, den är korrekt)
            for ticker, pos in list(positions.items()):
                if pos['shares'] > 0:
                    day_data = test_df[(test_df['date'] == date) & (test_df['ticker'] == ticker)]
                    if day_data.empty: continue
                    current_price = day_data['adj_close'].iloc[0]
                    should_sell, reason = False, ""
                    if current_price <= pos['purchase_price'] * 0.95:
                        should_sell, reason = True, "STOP-LOSS"
                    if not should_sell and ticker in models:
                        pipeline, le, best_features = models[ticker]
                        prediction = le.inverse_transform(pipeline.predict(day_data[best_features]))[0]
                        if prediction == 'Sälj': should_sell, reason = True, "SÄLJ-SIGNAL"
                    if should_sell:
                        sale_value = pos['shares'] * current_price
                        sale_fee = calculate_brokerage_fee(sale_value, brokerage_fixed_fee, brokerage_percentage)
                        current_capital += sale_value - sale_fee
                        total_brokerage_fees += sale_fee
                        total_transactions += 1
                        print(f"  - {str(date.date())}: {reason} {pos['shares']} st {ticker} @ {current_price:.2f}. Kapital: {current_capital:,.0f} kr")
                        positions[ticker] = {'shares': 0, 'purchase_price': 0}

            # Köplogik
            buy_signals_today = []
            for ticker in tickers:
                if ticker in models and positions[ticker]['shares'] == 0:
                    day_data = test_df[(test_df['date'] == date) & (test_df['ticker'] == ticker)]
                    if day_data.empty: continue
                    pipeline, le, best_features = models[ticker]
                    prediction = le.inverse_transform(pipeline.predict(day_data[best_features]))[0]
                    if prediction == 'Köp':
                        buy_signals_today.append({'ticker': ticker, 'price': day_data['adj_close'].iloc[0]})
            
            # ÄNDRAD KÖPLOGIK med kapitalkontroll
            if buy_signals_today and current_capital > brokerage_fixed_fee:
                capital_to_invest = current_capital * TRADE_CAPITAL_ALLOCATION
                capital_per_trade = capital_to_invest / len(buy_signals_today)
                for signal in buy_signals_today:
                    ticker, price = signal['ticker'], signal['price']
                    if price > 0:
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
            
            portfolio_value = current_capital
            for ticker, pos in positions.items():
                if pos['shares'] > 0:
                    last_price_data = prepared_dfs[ticker][prepared_dfs[ticker]['date'] <= date]
                    if not last_price_data.empty:
                        portfolio_value += pos['shares'] * last_price_data['adj_close'].iloc[-1]
            daily_portfolio_history.append({'date': date, 'portfolio_value': portfolio_value})
    
    if not daily_portfolio_history:
        print("Inga affärer gjordes under backtest-perioden.")
        return
        
    final_portfolio_value = daily_portfolio_history[-1]['portfolio_value']
    total_profit = final_portfolio_value - initial_capital
    history_df = pd.DataFrame(daily_portfolio_history).set_index('date')
    daily_returns = history_df['portfolio_value'].pct_change().dropna()
    sharpe, sortino, max_dd = calculate_sharpe_ratio(daily_returns), calculate_sortino_ratio(daily_returns), calculate_max_drawdown(history_df['portfolio_value'])
    buy_and_hold_capital = _calculate_portfolio_buy_and_hold(prepared_dfs, initial_capital, tickers, backtest_start_date, backtest_end_date)
    buy_and_hold_profit = buy_and_hold_capital - initial_capital


    print("\n" + "--- Slutresultat ---")
    print(f"Initialt kapital: {initial_capital:,.0f} kr")
    print(f"Slutkapital med portföljstrategi: {final_portfolio_value:,.0f} kr")
    print(f"Total vinst/förlust: {total_profit:,.0f} kr ({(total_profit/initial_capital)*100:.2f}%)")
    print(f"Total courtageavgift: {total_brokerage_fees:,.0f} kr")
    print(f"Totalt antal transaktioner: {total_transactions}")
    print("-" * 30)
    print(f"Sharpekvot (årlig): {sharpe:.2f}")
    print(f"Sortinokvot (årlig): {sortino:.2f}")
    print(f"Maximal nedgång (Drawdown): {max_dd:.2%}")
    print("\n" + "--- Jämförelse med Portfölj Köp & Behåll ---")
    print(f"Slutkapital med 'Köp och Behåll': {buy_and_hold_capital:,.0f} kr")
    print(f"Total vinst/förlust: {buy_and_hold_profit:,.0f} kr ({(buy_and_hold_profit/initial_capital)*100:.2f}%)")