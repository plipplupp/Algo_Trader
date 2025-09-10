import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from config import DatabaseConfig
import time

def run_grid_search_for_ticker(ticker, train_features, train_labels):
    """
    Söker efter optimala hyperparametrar för Random Forest-modellen med GridSearchCV.
    """
    start_time = time.time()
    print(f"\n--- Startar GridSearchCV för {ticker} ---")
    
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [10, 20, None]
    }
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(train_features, train_labels)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"GridSearchCV avslutad på {elapsed_time:.2f} sekunder.")
    print(f"Bästa parametrar: {grid_search.best_params_}")
    print(f"Bästa poäng (vägd F1): {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def run_backtest_for_ticker(ticker, initial_capital, brokerage_fee, days_to_predict, threshold):
    """
    Kör backtesting för en enskild ticker med maskininlärningsmodellen och en stop-loss.
    Jämför sedan resultatet med en "köp och behåll"-strategi.
    """
    db_name = DatabaseConfig.DB_NAME
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        
        query = f"SELECT * FROM stocks_prepared WHERE ticker = '{ticker}'"
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            print(f"Ingen data för {ticker}. Avbryter backtest.")
            return 0, initial_capital

        df.sort_values(by='date', inplace=True)
        
        # Skapa labels igen med de valda dagarna och tröskelvärdet
        df = create_future_label(df, days=days_to_predict, threshold=threshold)
        
        df.dropna(subset=['future_label'], inplace=True)

        features_df = df.drop(['ticker', 'future_label', 'date'], axis=1)
        features_df.fillna(0, inplace=True)

        labels = df['future_label']
        
        if features_df.empty or labels.isnull().all():
            print(f"Inga användbara rader kvar efter rensning för {ticker}. Avbryter backtest.")
            return 0, initial_capital

        split_index = int(len(features_df) * 0.8)
        train_features = features_df.iloc[:split_index]
        train_labels = labels.iloc[:split_index]
        test_features = features_df.iloc[split_index:]
        test_labels = labels.iloc[split_index:]

        if len(train_features) < 100 or len(test_features) < 50:
            print(f"För få rader för att köra backtest för {ticker}. Behöver mer data.")
            return 0, initial_capital

        print(f"\n--- Startar backtesting för {ticker} ---")
        print(f"Totala rader efter rensning: {len(features_df)}")
        print(f"Initialt kapital: {initial_capital:.2f} kr")
        print(f"Förutser {days_to_predict} dagar framåt med {threshold*100:.0f}% tröskel.")

        pipeline = run_grid_search_for_ticker(ticker, train_features, train_labels)
        predictions = pipeline.predict(test_features)
        
        print(f"\nKlassifikationsrapport på testdata för Random Forest:")
        print(classification_report(test_labels, predictions, zero_division=0))
        
        # Simuleringskoden med stop-loss
        current_capital = initial_capital
        position = 0
        position_size = 0
        purchase_price = 0
        stop_loss_percent = 0.05  # 5% stop-loss
        
        print("\nSimulerar handel med stop-loss...")
        backtest_df = df.iloc[split_index:].copy()
        backtest_df['prediction'] = predictions
        
        for i in range(len(backtest_df)):
            prediction = backtest_df['prediction'].iloc[i]
            current_price = backtest_df['adj_close'].iloc[i]
            
            # KONTROLL 1: Stop-loss
            if position == 1 and current_price <= purchase_price * (1 - stop_loss_percent):
                sale_value = position_size * current_price
                final_sale_value = sale_value - (brokerage_fee if brokerage_fee > 1 else (brokerage_fee / 100) * sale_value)
                current_capital = final_sale_value
                position = 0
                print(f"Stop-loss triggad! Position såld på {current_price:.2f} kr. Kapital: {current_capital:.2f} kr")
            
            # KONTROLL 2: Modellens sälj-signal
            elif prediction == 'Sälj' and position == 1:
                sale_value = position_size * current_price
                final_sale_value = sale_value - (brokerage_fee if brokerage_fee > 1 else (brokerage_fee / 100) * sale_value)
                current_capital = final_sale_value
                position = 0
            
            # KONTROLL 3: Modellens köp-signal
            if prediction == 'Köp' and position == 0:
                if current_capital > 0:
                    position_size = current_capital / current_price
                    purchase_price = current_price  # Spara köppriset för stop-loss
                    current_capital = current_capital - (brokerage_fee if brokerage_fee > 1 else (brokerage_fee / 100) * current_capital)
                    position = 1

        if position == 1:
            current_capital = position_size * backtest_df['adj_close'].iloc[-1]
        
        profit = current_capital - initial_capital
        print(f"\nSlutkapital med strategi: {current_capital:.2f} kr")
        print(f"Total vinst/förlust: {profit:.2f} kr ({ (profit / initial_capital) * 100:.2f}%)")
        
        # Analys av "Köp och Behåll"-strategin
        buy_and_hold_return = calculate_buy_and_hold_return(backtest_df, initial_capital)
        print(f"Slutkapital med 'Köp och Behåll': {buy_and_hold_return['final_capital']:.2f} kr")
        print(f"Total vinst/förlust: {buy_and_hold_return['profit']:.2f} kr ({buy_and_hold_return['percent_change']:.2f}%)")
        print("--------------------------------------------------------------------")
        
        return profit, current_capital

    except sqlite3.Error as e:
        print(f"Ett fel uppstod: {e}")
        return 0, initial_capital
    finally:
        if conn:
            conn.close()

def create_future_label(df, days, threshold):
    """
    Skapar målvariabeln (label) baserat på framtida prisrörelse.
    """
    if df.empty or 'adj_close' not in df.columns:
        return df
        
    price_future = df['adj_close'].shift(-days)
    price_change = (price_future - df['adj_close']) / df['adj_close']
    
    df['future_label'] = 'Behåll'
    df.loc[price_change >= threshold, 'future_label'] = 'Köp'
    df.loc[price_change <= -threshold, 'future_label'] = 'Sälj'
    
    df.loc[df.index.isin(df.index[-days:]), 'future_label'] = None
    
    return df

def calculate_buy_and_hold_return(df, initial_capital):
    """
    Beräknar avkastningen för en "köp och behåll"-strategi.
    """
    if df.empty:
        return {'profit': 0, 'percent_change': 0, 'final_capital': initial_capital}
        
    initial_price = df['adj_close'].iloc[0]
    final_price = df['adj_close'].iloc[-1]
    
    # Beräkna hur många aktier som kunde köpas från startkapitalet
    shares_held = initial_capital / initial_price
    
    # Beräkna slutkapitalet
    final_capital = shares_held * final_price
    
    profit = final_capital - initial_capital
    percent_change = (profit / initial_capital) * 100
    
    return {'profit': profit, 'percent_change': percent_change, 'final_capital': final_capital}