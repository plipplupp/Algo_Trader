import os
from datetime import datetime

# Funktion för att läsa in tickers från en fil
def load_tickers_from_file(filename):
    """
    Läser en lista med tickers från en textfil.
    Varje ticker ska stå på en separat rad.
    """
    tickers = []
    filepath = os.path.join(os.path.dirname(__file__), filename)
    try:
        with open(filepath, 'r') as file:
            for line in file:
                ticker = line.strip()
                if ticker:
                    tickers.append(ticker)
    except FileNotFoundError:
        print(f"Fel: Filen '{filename}' hittades inte.")
        return []
    return tickers

# Konfiguration för datahämtning och tickers
class DataConfig:
    """Inställningar för att hämta aktiedata."""
    TICKERS = load_tickers_from_file("tickers.txt")
    START_DATE = "2020-01-01"
    END_DATE = datetime.now().strftime('%Y-%m-%d')


# Konfiguration för databasen
class DatabaseConfig:
    """Inställningar för databasanslutningen."""
    # Uppdaterat namn på databasfilen
    DB_NAME = "stock_data.db"


# Konfiguration för backtesting
class BacktestConfig:
    """Inställningar för backtesting-strategin."""
    INITIAL_CAPITAL = 100000
    BROKERAGE_FIXED_FEE = 69.0
    BROKERAGE_PERCENTAGE = 0.00069

# Konfiguration för optimala parametrar för varje ticker
class OptimalParamsConfig:
    """Optimala parametrar för varje ticker."""    
    OPTIMAL_PARAMS = {
        'AAPL': {'days': 5, 'threshold': 0.03},
        'GOOGL': {'days': 5, 'threshold': 0.03},
        'MSFT': {'days': 5, 'threshold': 0.03},
        'NVDA': {'days': 10, 'threshold': 0.04},
        'SWED-A.ST': {'days': 15, 'threshold': 0.04},
        'TSLA': {'days': 10, 'threshold': 0.02}
    }
    

# Konfiguration för API-nycklar (exempel för framtida behov)
class ApiConfig:
    """Inställningar för externa API-nycklar."""
    pass


# Konfiguration för modellträning (framtida behov)
class ModelConfig:
    """Inställningar för den neurala nätverksmodellen."""
    pass
