import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
import math
from config import DatabaseConfig
from utils import calculate_brokerage_fee, create_future_label

def select_best_features(features_df, labels, top_n=60):
    model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
    sample_weights = compute_sample_weight(class_weight='balanced', y=labels)
    model.fit(features_df, labels, sample_weight=sample_weights)
    
    feature_importances = pd.Series(model.feature_importances_, index=features_df.columns)
    best_features = feature_importances.nlargest(top_n).index.tolist()
    return best_features

def run_walk_forward_optimization_for_ticker(ticker, initial_capital, brokerage_fixed_fee, brokerage_percentage, days_to_predict, threshold):
    db_name = DatabaseConfig.DB_NAME
    try:
        conn = sqlite3.connect(db_name)
        df = pd.read_sql_query(f"SELECT * FROM stocks_prepared WHERE ticker = '{ticker}'", conn)
    finally:
        if conn: conn.close()

    if df.empty or len(df) < 252:
        return 0, {}, 0, None, None

    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    df = create_future_label(df, days=days_to_predict, threshold=threshold)
    df.dropna(subset=['future_label'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    if df.empty: return 0, {}, 0, None, None

    features_df = df.drop(columns=['ticker', 'future_label', 'date']).replace([np.inf, -np.inf], 0).fillna(0)
    labels = df['future_label']

    le = LabelEncoder()
    numeric_labels = le.fit_transform(labels)
    
    tscv = TimeSeriesSplit(n_splits=4)
    total_profit = 0
    last_report = {}
    full_backtest_start_date, full_backtest_end_date = None, None
    total_days_in_backtest = 0
    
    # NYTT: Behållare för handelsloggar
    full_trade_log = []

    for fold_count, (train_index, test_index) in enumerate(tscv.split(features_df), 1):
        print(f"  - Fold {fold_count}/{tscv.get_n_splits()}: Tränar på {len(train_index)} rader, testar på {len(test_index)} rader.")

        train_features_all, train_labels_numeric = features_df.iloc[train_index], numeric_labels[train_index]
        
        if len(np.unique(train_labels_numeric)) < 2:
            print("     - Skippar fold, endast en klass i träningsdatan.")
            continue
            
        best_features = select_best_features(train_features_all, train_labels_numeric)
        train_features_selected = train_features_all[best_features]
        test_features_selected = features_df.iloc[test_index][best_features]

        if full_backtest_start_date is None: full_backtest_start_date = df.loc[test_index[0], 'date']
        full_backtest_end_date = df.loc[test_index[-1], 'date']
        total_days_in_backtest += len(test_index)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'))
        ])
        
        sample_weights = compute_sample_weight(class_weight='balanced', y=train_labels_numeric)
        pipeline.fit(train_features_selected, train_labels_numeric, classifier__sample_weight=sample_weights)

        predictions_numeric = pipeline.predict(test_features_selected)
        predictions_text = le.inverse_transform(predictions_numeric)
        test_labels_text = le.inverse_transform(numeric_labels[test_index])
        
        last_report = classification_report(test_labels_text, predictions_text, labels=le.classes_, zero_division=0, output_dict=True)

        # === NY, KORREKT SIMULERINGSLOGIK FÖR VARJE FOLD ===
        fold_capital = initial_capital # Återställ kapitalet för varje fold
        position_size = 0
        purchase_price = 0
        
        backtest_df_fold = df.iloc[test_index].copy()
        backtest_df_fold['prediction'] = predictions_text

        for _, row in backtest_df_fold.iterrows():
            # Säljlogik
            if position_size > 0 and (row['prediction'] == 'Sälj' or row['adj_close'] <= purchase_price * 0.95):
                sale_value = position_size * row['adj_close']
                sale_fee = calculate_brokerage_fee(sale_value, brokerage_fixed_fee, brokerage_percentage)
                capital_before = fold_capital
                fold_capital += sale_value - sale_fee
                full_trade_log.append({'date': row['date'], 'action': 'SELL', 'price': row['adj_close'], 'shares': position_size, 'capital': fold_capital})
                position_size = 0
            
            # Köplogik (endast en position i taget, använder en del av kapitalet)
            if position_size == 0 and row['prediction'] == 'Köp':
                price = row['adj_close']
                if price > 0 and fold_capital > 100: # Säkerhetsmarginal
                    # Använd 95% av kapitalet för att säkerställa att courtage täcks
                    capital_to_use = fold_capital * 0.95 
                    shares_to_buy = math.floor(capital_to_use / price)
                    if shares_to_buy > 0:
                        buy_cost = (shares_to_buy * price) + calculate_brokerage_fee(shares_to_buy * price, brokerage_fixed_fee, brokerage_percentage)
                        if buy_cost <= fold_capital:
                            position_size = shares_to_buy
                            purchase_price = price
                            fold_capital -= buy_cost
                            full_trade_log.append({'date': row['date'], 'action': 'BUY', 'price': price, 'shares': shares_to_buy, 'capital': fold_capital})
        
        # Avsluta eventuell öppen position i slutet av folden och beräkna vinsten
        if position_size > 0:
            final_price = backtest_df_fold.iloc[-1]['adj_close']
            fold_capital += (position_size * final_price) - calculate_brokerage_fee(position_size * final_price, brokerage_fixed_fee, brokerage_percentage)
        
        fold_profit = fold_capital - initial_capital
        total_profit += fold_profit # Summera vinsten från varje oberoende fold

    # NYTT: Skriv ut handelsloggen om resultatet blev negativt
    if total_profit < 0:
        print("\n--- DETALJERAD HANDELSLOGG FÖR FÖRLUSTSTRATEGI ---")
        log_df = pd.DataFrame(full_trade_log)
        if not log_df.empty:
            log_df['date'] = log_df['date'].dt.date
            print(log_df.to_string())
        else:
            print("Inga affärer gjordes.")
        print("--------------------------------------------------\n")

    return total_profit, last_report, total_days_in_backtest, full_backtest_start_date, full_backtest_end_date

def run_backtest_for_ticker(ticker, initial_capital, brokerage_fixed_fee, brokerage_percentage, days, threshold):
    profit, report, days_in_backtest, start_date, end_date = run_walk_forward_optimization_for_ticker(ticker, initial_capital, brokerage_fixed_fee, brokerage_percentage, days, threshold)
    return profit, None, report, days_in_backtest, start_date, end_date