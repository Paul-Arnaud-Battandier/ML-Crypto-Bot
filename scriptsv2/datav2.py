import pandas as pd
import requests
import time

def fetch_binance_1m_data(symbol="BTCUSDT", limit=1000, loops=10):
    """
    Récupère l'historique 1-minute sur Binance.
    Par défaut : 10 boucles de 1000 bougies = 10 000 minutes (~7 jours).
    """
    print(f"⏳ Téléchargement des données {symbol} (1 minute)...")
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    end_time = None
    
    for i in range(loops):
        params = {
            "symbol": symbol,
            "interval": "1m",
            "limit": limit
        }
        # Si on a déjà récupéré des données, on demande à l'API de reculer dans le temps
        if end_time:
            params["endTime"] = end_time
            
        response = requests.get(url, params=params)
        data = response.json()
        
        if not data:
            break
            
        # On ajoute les nouvelles données au début de notre liste
        all_data = data + all_data 
        
        # Le nouveau point d'arrêt (end_time) est juste avant la plus vieille bougie récupérée
        end_time = data[0][0] - 1 
        
        # On fait une pause de 0.5s pour ne pas se faire bannir par l'API Binance
        time.sleep(0.5) 
        
    # Formatage propre
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
               'close_time', 'quote_asset_volume', 'number_of_trades', 
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    
    df = pd.DataFrame(all_data, columns=columns)
    
    # Gestion du temps : Binance renvoie des millisecondes (unit='ms')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    
    # On ne garde que l'essentiel et on force le format nombre
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    
    # Sécurité anti-doublons
    df = df[~df.index.duplicated(keep='first')].sort_index()
    
    print(f"✅ Succès : {len(df)} lignes récupérées.")
    print(f"📅 Du : {df.index.min()}")
    print(f"📅 Au : {df.index.max()}")
    
    return df

if __name__ == "__main__":
    # 1. Extraction
    df_raw = fetch_binance_1m_data(loops=10) # Tu peux augmenter loops=30 pour 21 jours
    
    # 2. Sauvegarde
    output_path = "data/binance_1m_raw.csv"
    df_raw.to_csv(output_path)
    print(f"💾 Données brutes sauvegardées dans '{output_path}'")