# config.py

import os

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


# Konfiguration för databasen
class DatabaseConfig:
    """Inställningar för databasanslutningen."""
    # Uppdaterat namn på databasfilen
    DB_NAME = "stock_data.db"


# Konfiguration för API-nycklar (exempel för framtida behov)
class ApiConfig:
    """Inställningar för externa API-nycklar."""
    pass


# Konfiguration för modellträning (framtida behov)
class ModelConfig:
    """Inställningar för den neurala nätverksmodellen."""
    pass