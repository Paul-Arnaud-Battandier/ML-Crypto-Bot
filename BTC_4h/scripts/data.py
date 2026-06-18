import pandas as pd
import numpy as np
import time
import yfinance as yf
import ccxt
import requests
from config import SYMBOL, SYMBOL_FUTURES, TIMEFRAME, START_DATE, END_DATE, DATA_RAW, BINANCE_FUTURES_URL


def fetch_ohlcv_raw(symbol, timeframe, start_date, end_date):
    # 1. Setup
    exchange = ccxt.binance()
    # Conversion en ms (format entier brut)
    since = pd.Timestamp(start_date).value // 10**6
    end_timestamp = pd.Timestamp(end_date).value // 10**6
    
    all_candles = []

    # 2. Boucle d'extraction (Pagination CCXT)
    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not candles:
                break
            
            all_candles.extend(candles)
            since = candles[-1][0] + 1 # On avance au timestamp suivant
            
            if candles[-1][0] >= end_timestamp:
                break
                
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"Extraction error: {e}")
            break

    # 3. Retourne le DataFrame brut avec colonnes nommées
    return pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])


def fetch_funding_rates_raw(symbol, start_date, end_date):
    # 1. Conversion minimale pour l'API
    since = pd.Timestamp(start_date).value // 10**6
    end_timestamp = pd.Timestamp(end_date).value // 10**6
    all_data = []

    # 2. Boucle d'extraction
    while True:
        params = {
            "symbol": symbol,
            "startTime": since,
            "limit": 1000
        }

        response = requests.get(BINANCE_FUTURES_URL, params=params)
        response.raise_for_status()
        data = response.json()

        if not data:
            break

        all_data.extend(data)
        
        # On avance le curseur via le dernier timestamp reçu
        since = data[-1]["fundingTime"] + 1

        if since > end_timestamp:
            break

        time.sleep(0.5) # Respect du rate limit

    # 3. Retourne le DF brut tel quel
    return pd.DataFrame(all_data)


def fetch_fear_greed_raw():
    # On prend une limite large pour couvrir ton START_DATE
    response = requests.get("https://api.alternative.me/fng/?limit=3000")
    response.raise_for_status()
    data = response.json()
    
    # On retourne le DataFrame brut contenu dans la clé 'data'
    return pd.DataFrame(data["data"])


def fetch_macro_raw(start_date, end_date):
    # On télécharge les deux tickers simultanément
    # yfinance renvoie un DataFrame avec un MultiIndex pour les colonnes
    df = yf.download(["DX-Y.NYB", "QQQ"], start=start_date, end=end_date, auto_adjust=True)
    return df


def main():

    # 1. fetch OHLCV
    df_ohlcv = fetch_ohlcv_raw(SYMBOL_FUTURES, TIMEFRAME, START_DATE, END_DATE)

    # 2. fetch funding rates
    df_funding = fetch_funding_rates_raw(SYMBOL_FUTURES, START_DATE, END_DATE)

    # 3. fetch fear & greed
    df_fng = fetch_fear_greed_raw()

    # 4. fetch macro
    df_macro = fetch_macro_raw(START_DATE, END_DATE)

    df_ohlcv.to_csv("data/raw/ohlcv_raw.csv", index=False)
    df_funding.to_csv("data/raw/funding_raw.csv", index=False)
    df_fng.to_csv("data/raw/fng_raw.csv", index=False)
    df_macro.to_csv("data/raw/macro_raw.csv")


#python -m scripts.data
if __name__ == "__main__":
    main()