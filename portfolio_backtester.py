import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import math
from config import DatabaseConfig

def calculate_brokerage_fee(transaction_cost, fixed_fee, percentage_fee):
    """
    Beräknar courtageavgiften baserat på transaktionskostnaden.
    Använder en tiered-modell: minst en fast avgift eller en procentuell avgift av kostnaden.
    """
    calculated_percentage_fee = transaction_cost * percentage_fee
    return max(fixed_fee, calculated_percentage_fee)

def run_portfolio_backtest(tickers, optimal_params, initial_capital, brokerage_fixed_fee, brokerage_percentage):
    """
    Kör backtesting för en hel portfölj av aktier med de optimala parametrarna.
    Allokerar kapital dynamiskt baserat på modellens köpsignaler.
    """
    db_name = DatabaseConfig.DB_NAME
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        
        all_data = []
        train_test_split_dates = {}
        total_brokerage_fees = 0.0  # Ny variabel för totala avgifter
        total_transactions = 0      # Ny räknare för transaktioner

        # 1. Ladda data och träna modeller för varje ticker
        print("--- Laddar data och tränar modeller för varje aktie ---")
        for ticker in tickers:
            query = f"SELECT * FROM stocks_prepared WHERE ticker = '{ticker}'"
            df = pd.read_sql_query(query, conn)
            
            if df.empty:
                print(f"Varning: Ingen data för {ticker}. Skippar.")
                continue
            
            # Använd optimala parametrar från analysen
            params = optimal_params.get(ticker, {'days': 10, 'threshold': 0.03})
            days_to_predict = params['days']
            threshold = params['threshold']
            
            df = create_future_label(df, days=days_to_predict, threshold=threshold)
            df.dropna(subset=['future_label'], inplace=True)

            # Spara datum för testperioden
            split_index = int(len(df) * 0.8)
            train_test_split_dates[ticker] = df.iloc[split_index]['date']

            features_df = df.drop(['ticker', 'future_label', 'date'], axis=1)
            features_df.fillna(0, inplace=True)
            labels = df['future_label']
            
            if len(features_df) < 50:
                print(f"Varning: För få rader för {ticker}. Skippar.")
                continue

            train_features = features_df.iloc[:split_index]
            train_labels = labels.iloc[:split_index]
            test_features = features_df.iloc[split_index:]
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100, max_depth=20))
            ])
            
            pipeline.fit(train_features, train_labels)
            predictions = pipeline.predict(test_features)
            
            df_test = df.iloc[split_index:].copy()
            df_test['prediction'] = predictions
            df_test['ticker'] = ticker
            
            all_data.append(df_test)
            print(f"Modell tränad och redo för {ticker}.")
            
        if not all_data:
            print("Inget backtest kunde utföras.")
            return

        # 2. Sammanfoga all testdata till en portfölj-data frame
        portfolio_df = pd.concat(all_data)
        portfolio_df.sort_values(by='date', inplace=True)
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        
        print("\n--- Startar Portfölj-backtest ---")
        
        # 3. Simuleringslogik
        current_capital = initial_capital
        positions = {}  # {'AAPL': {'shares': 0, 'purchase_price': 0}}
        daily_capital = []
        
        dates = portfolio_df['date'].unique()
        
        for date in dates:
            day_data = portfolio_df[portfolio_df['date'] == date]
            
            # Stäng positioner baserat på 'Sälj' eller stop-loss
            tickers_to_sell = []
            for ticker, pos in list(positions.items()):
                # Kontrollera att data för tickern existerar för den aktuella dagen
                ticker_day_data = day_data[day_data['ticker'] == ticker]
                if ticker_day_data.empty:
                    continue # Hoppa över om ingen data finns för denna dag

                current_price = ticker_day_data['adj_close'].iloc[0]
                purchase_price = pos['purchase_price']
                
                # Beräkna säljavgiften dynamiskt
                sell_cost = pos['shares'] * current_price
                sell_fee = calculate_brokerage_fee(sell_cost, brokerage_fixed_fee, brokerage_percentage)

                # Stop-loss
                if current_price <= purchase_price * (1 - 0.05): # 5% stop-loss
                    profit_loss = (current_price - purchase_price) * pos['shares']
                    current_capital += sell_cost - sell_fee
                    total_brokerage_fees += sell_fee  # Addera avgiften
                    total_transactions += 1 # Inkrementera transaktionsräknaren
                    print(f"Stop-loss: sålde {ticker} på {date.date()} för {current_price:.2f} kr. Vinst/förlust: {profit_loss:.2f} kr")
                    tickers_to_sell.append(ticker)
                
                # Sälj-signal från modellen
                elif ticker_day_data['prediction'].iloc[0] == 'Sälj':
                    profit_loss = (current_price - purchase_price) * pos['shares']
                    current_capital += sell_cost - sell_fee
                    total_brokerage_fees += sell_fee  # Addera avgiften
                    total_transactions += 1 # Inkrementera transaktionsräknaren
                    print(f"Sälj-signal: sålde {ticker} på {date.date()} för {current_price:.2f} kr. Vinst/förlust: {profit_loss:.2f} kr")
                    tickers_to_sell.append(ticker)

            for ticker in tickers_to_sell:
                del positions[ticker]
            
            # Öppna nya positioner baserat på 'Köp'
            buy_signals = day_data[day_data['prediction'] == 'Köp']
            if not buy_signals.empty and current_capital > 0:
                # Justera denna variabel för att ändra hur stor del av det tillgängliga kapitalet som ska allokeras per signal
                fraction_per_signal = 0.95
                fraction_of_capital = 1 / len(buy_signals) * fraction_per_signal
                
                for _, row in buy_signals.iterrows():
                    ticker = row['ticker']
                    price = row['adj_close']
                    
                    if ticker not in positions and current_capital > 0:
                      investment_capital = current_capital * fraction_of_capital
                      
                      # Starta med det teoretiska max-antalet aktier
                      shares_to_buy = math.floor(investment_capital / price)
                      
                      # Iterera nedåt tills den totala kostnaden (inklusive courtage) ryms
                      while shares_to_buy > 0:
                          buy_cost_no_fee = shares_to_buy * price
                          buy_fee = calculate_brokerage_fee(buy_cost_no_fee, brokerage_fixed_fee, brokerage_percentage)
                          total_cost = buy_cost_no_fee + buy_fee
                          
                          if total_cost <= investment_capital:
                              # Vi har hittat ett antal aktier som fungerar, avbryt loopen
                              break
                          
                          # Om kostnaden är för hög, minska med en aktie och försök igen
                          shares_to_buy -= 1

                      if shares_to_buy > 0:
                          # Nu vet vi att total_cost (från loopen) är korrekt och ryms inom budgeten
                          cost = total_cost
                          positions[ticker] = {'shares': shares_to_buy, 'purchase_price': price}
                          current_capital -= cost
                          total_brokerage_fees += buy_fee  # Addera avgiften
                          total_transactions += 1 # Inkrementera transaktionsräknaren
                          print(f"Köp-signal: köpte {shares_to_buy} st {ticker} på {date.date()} för {price:.2f} kr")
                      # Om shares_to_buy är 0 efter loopen, var inte ens en aktie + courtage möjlig
                      # Då kan vi antingen skriva ut ett meddelande eller bara låta bli att handla tyst.
                      elif investment_capital > 0: # Skriv bara ut om det faktiskt fanns ett försök
                           print(f"** Courtageblockering för {ticker} på {date.date()}: Inte ens en aktie kunde köpas inom budgeten ({investment_capital:.2f} kr). **")            

            
            # Uppdatera det dagliga kapitalet (summan av kontanter och aktievärde)
            total_portfolio_value = current_capital
            for ticker, pos in positions.items():
                ticker_day_data = day_data[day_data['ticker'] == ticker]
                if not ticker_day_data.empty:
                    last_price = ticker_day_data['adj_close'].iloc[0]
                    total_portfolio_value += pos['shares'] * last_price
            
            daily_capital.append({'date': date, 'capital': total_portfolio_value})

        # Slutlig summering
        final_capital = current_capital
        for ticker, pos in positions.items():
            last_price_df = portfolio_df[portfolio_df['ticker'] == ticker]
            if not last_price_df.empty:
                last_price = last_price_df['adj_close'].iloc[-1]
                final_capital += pos['shares'] * last_price
        
        profit = final_capital - initial_capital
        
        print("\n--- Slutresultat ---")
        print(f"Initialt kapital: {initial_capital:.2f} kr")
        print(f"Slutkapital med portföljstrategi: {final_capital:.2f} kr")
        print(f"Total vinst/förlust: {profit:.2f} kr ({ (profit / initial_capital) * 100:.2f}%)")
        print(f"Total courtageavgift: {total_brokerage_fees:.2f} kr")
        print(f"Totalt antal transaktioner: {total_transactions}")

        # Jämförelse med "Köp och Behåll" för hela portföljen
        buy_and_hold_profit, buy_and_hold_capital = calculate_portfolio_buy_and_hold(portfolio_df, initial_capital, tickers)
        print("\n--- Jämförelse med Portfölj Köp & Behåll ---")
        print(f"Slutkapital med 'Köp och Behåll': {buy_and_hold_capital:.2f} kr")
        print(f"Total vinst/förlust: {buy_and_hold_profit:.2f} kr ({ (buy_and_hold_profit / initial_capital) * 100:.2f}%)")
        
    except sqlite3.Error as e:
        print(f"Ett fel uppstod: {e}")
    finally:
        if conn:
            conn.close()

def create_future_label(df, days, threshold):
    """ Skapar målvariabeln (label) baserat på framtida prisrörelse. """
    if df.empty or 'adj_close' not in df.columns:
        return df
        
    price_future = df['adj_close'].shift(-days)
    price_change = (price_future - df['adj_close']) / df['adj_close']
    
    df['future_label'] = 'Behåll'
    df.loc[price_change >= threshold, 'future_label'] = 'Köp'
    df.loc[price_change <= -threshold, 'future_label'] = 'Sälj'
    
    df.loc[df.index.isin(df.index[-days:]), 'future_label'] = None
    
    return df

def calculate_portfolio_buy_and_hold(df, initial_capital, tickers):
    """ Beräknar avkastningen för en "köp och behåll"-strategi för hela portföljen. """
    shares_per_ticker = initial_capital / len(tickers)
    total_profit = 0
    
    for ticker in tickers:
        ticker_df = df[df['ticker'] == ticker]
        if ticker_df.empty:
            continue
            
        initial_price = ticker_df['adj_close'].iloc[0]
        final_price = ticker_df['adj_close'].iloc[-1]
        
        shares_bought = shares_per_ticker / initial_price
        profit = (final_price - initial_price) * shares_bought
        total_profit += profit
        
    final_capital = initial_capital + total_profit
    
    return total_profit, final_capital
