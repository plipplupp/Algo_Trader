import sqlite3
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from config import DatabaseConfig
import math
from utils import calculate_brokerage_fee, create_future_label

def run_walk_forward_optimization_for_ticker(ticker, initial_capital, brokerage_fixed_fee, brokerage_percentage, days_to_predict, threshold):
    db_name = DatabaseConfig.DB_NAME
    try:
        conn = sqlite3.connect(db_name)
        df = pd.read_sql_query(f"SELECT * FROM stocks_prepared WHERE ticker = '{ticker}'", conn)
    finally:
        if conn:
            conn.close()

    if df.empty or len(df) < 100:
        return 0, {}, 0, None, None

    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    df = create_future_label(df, days=days_to_predict, threshold=threshold)
    df.dropna(subset=['future_label'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    features_df = df.drop(columns=['ticker', 'future_label', 'date'])
    features_df.fillna(0, inplace=True)
    labels = df['future_label']

    le = LabelEncoder()
    numeric_labels = le.fit_transform(labels)
    
    n_splits = 4 
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    current_capital = initial_capital
    last_report = {}
    
    full_backtest_start_date, full_backtest_end_date = None, None
    total_days_in_backtest = 0

    fold_count = 0
    for train_index, test_index in tscv.split(features_df):
        fold_count += 1
        print(f"  - Fold {fold_count}/{n_splits}: Tränar på {len(train_index)} rader, testar på {len(test_index)} rader.")

        train_features, train_labels = features_df.iloc[train_index], numeric_labels[train_index]
        test_features, test_labels_numeric = features_df.iloc[test_index], numeric_labels[test_index]
        
        if full_backtest_start_date is None:
            full_backtest_start_date = df.loc[test_index[0], 'date']
        full_backtest_end_date = df.loc[test_index[-1], 'date']
        total_days_in_backtest += len(test_index)

        ### FIX: Tar bort "use_label_encoder=False" som inte längre används.
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'))
        ])
        pipeline.fit(train_features, train_labels)

        predictions_numeric = pipeline.predict(test_features)
        predictions_text = le.inverse_transform(predictions_numeric)
        test_labels_text = le.inverse_transform(test_labels_numeric)
        
        last_report = classification_report(test_labels_text, predictions_text, labels=le.classes_, zero_division=0, output_dict=True)

        backtest_df_fold = df.iloc[test_index].copy()
        backtest_df_fold['prediction'] = predictions_text
        
        position_size = 0
        purchase_price = 0

        for _, row in backtest_df_fold.iterrows():
            if position_size > 0 and (row['prediction'] == 'Sälj' or row['adj_close'] <= purchase_price * 0.95):
                sale_cost = position_size * row['adj_close']
                sale_fee = calculate_brokerage_fee(sale_cost, brokerage_fixed_fee, brokerage_percentage)
                current_capital += sale_cost - sale_fee
                position_size = 0
            
            if position_size == 0 and row['prediction'] == 'Köp':
                price = row['adj_close']
                if price > 0:
                    capital_to_use = current_capital
                    shares_to_buy = math.floor(capital_to_use / price)
                    while shares_to_buy > 0:
                        buy_cost = (shares_to_buy * price) + calculate_brokerage_fee(shares_to_buy * price, brokerage_fixed_fee, brokerage_percentage)
                        if buy_cost <= capital_to_use:
                            break
                        shares_to_buy -= 1
                    if shares_to_buy > 0:
                        buy_cost = (shares_to_buy * price) + calculate_brokerage_fee(shares_to_buy * price, brokerage_fixed_fee, brokerage_percentage)
                        position_size = shares_to_buy
                        purchase_price = price
                        current_capital -= buy_cost

    if position_size > 0:
        final_price = df.iloc[-1]['adj_close']
        current_capital += (position_size * final_price) - calculate_brokerage_fee(position_size * final_price, brokerage_fixed_fee, brokerage_percentage)
        
    total_profit = current_capital - initial_capital

    return total_profit, last_report, total_days_in_backtest, full_backtest_start_date, full_backtest_end_date

def run_backtest_for_ticker(ticker, initial_capital, brokerage_fixed_fee, brokerage_percentage, days, threshold):
    profit, report, days_in_backtest, start_date, end_date = run_walk_forward_optimization_for_ticker(ticker, initial_capital, brokerage_fixed_fee, brokerage_percentage, days, threshold)
    return profit, None, report, days_in_backtest, start_date, end_date