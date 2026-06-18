import requests
import pandas as pd
import time

def fetch_binance_1m_data(symbol="BTCUSDT", loops=1):
    """
    Télécharge les données depuis l'API publique 'Vision' de Binance 
    (immunisée contre la plupart des blocages IP Cloud).
    """
    # L'URL magique pour les serveurs Cloud :
    url = "https://data-api.binance.vision/api/v3/klines"
    
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
            
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        # --- BOUCLIER CLOUD ---
        # Si Binance renvoie un dictionnaire (une erreur) au lieu d'une liste
        if isinstance(data, dict):
            raise ValueError(f"Binance a bloqué la requête. Message API : {data}")
        # ----------------------
        
        if not data:
            break
            
        all_klines = data + all_klines
        end_time = data[0][0] - 1
        time.sleep(0.1)
        
    # Formatage classique du DataFrame
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