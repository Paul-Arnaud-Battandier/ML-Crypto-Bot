import requests
import pandas as pd
import time

def fetch_binance_1m_data(symbol="BTCUSDT", loops=1):
    """
    Télécharge les données depuis l'API Binance Futures (fapi).
    Le contournement parfait contre les blocages d'IP partagées des hébergeurs Cloud,
    car les serveurs Futures ont des quotas séparés du marché Spot.
    """
    # L'URL "Porte de derrière" (Binance Futures)
    url = "https://fapi.binance.com/fapi/v1/klines"
    
    all_klines = []
    end_time = None
    
    for _ in range(loops):
        params = {
            "symbol": symbol,
            "interval": "1m",
            "limit": 1000
        }
        if end_time:
            params["endTime"] = end_time
            
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            # --- BOUCLIER CLOUD ---
            if isinstance(data, dict):
                raise ValueError(f"Blocage Binance Futures détecté : {data}")
            # ----------------------
            
            if not data:
                break
                
            all_klines = data + all_klines
            end_time = data[0][0] - 1
            time.sleep(0.1)
            
        except Exception as e:
            print(f"⚠️ Avertissement API : {e}")
            # Si ça échoue, on lève une exception pour que le main.py 
            # annule le trade de cette minute sans faire crasher le bot entier.
            raise e
        
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
               'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
               'taker_buy_quote_asset_volume', 'ignore']
    
    df = pd.DataFrame(all_klines, columns=columns)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
        
    return df

if __name__ == "__main__":
    # 1. Extraction
    df_raw = fetch_binance_1m_data(loops=10) # Tu peux augmenter loops=30 pour 21 jours
    
    # 2. Sauvegarde
    output_path = "data/binance_1m_raw.csv"
    df_raw.to_csv(output_path)
    print(f"💾 Données brutes sauvegardées dans '{output_path}'")