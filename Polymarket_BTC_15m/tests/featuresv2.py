import pandas as pd
import numpy as np
import ta  # Librairie d'analyse technique (pip install ta)

def create_lgbm_features(input_csv, output_csv):
    print(f"🧠 Chargement des données brutes depuis {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Assurons-nous que le timestamp est au bon format et trié
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print("⚙️ Calcul des Fenêtres Glissantes (Mémoire du Marché)...")

    # 1. MOMENTUM (Vitesse du prix)
    # Quelle a été la variation du prix sur les 5 et 15 dernières minutes ?
    df['return_5m'] = df['close'].pct_change(periods=5)
    df['return_15m'] = df['close'].pct_change(periods=15)
    
    # 2. VOLATILITÉ (Risque et Chaos)
    # L'écart-type des rendements sur la dernière heure (60 minutes)
    df['volatility_60m'] = df['close'].pct_change().rolling(window=60).std()
    
    # 3. VOLUME SURGE (Pression institutionnelle)
    # Le volume des 15 dernières minutes est-il anormalement élevé comparé aux 4 dernières heures ?
    rolling_vol_15m = df['volume'].rolling(window=15).sum()
    rolling_vol_4h = df['volume'].rolling(window=240).mean() * 15 # Moyenne ajustée sur 15m
    df['volume_surge'] = rolling_vol_15m / rolling_vol_4h
    
    # 4. INDICATEURS TECHNIQUES (RSI)
    # On utilise la librairie 'ta' pour calculer le RSI sur 14 périodes
    df['rsi_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()

    # 5. LA TARGET (Ce que le modèle doit deviner)
    # Le prix sera-t-il plus haut dans exactement 15 minutes ? (1 = UP, 0 = DOWN)
    # On décale la colonne 'close' de -15 pour avoir le prix futur sur la ligne actuelle
    df['future_close_15m'] = df['close'].shift(-15)
    df['target'] = (df['future_close_15m'] > df['close']).astype(int)

    print("🧹 Nettoyage des données incomplètes (NaN)...")
    # On supprime les premières lignes qui n'ont pas assez d'historique 
    # et les dernières qui n'ont pas de futur (target)
    df = df.dropna().copy()
    
    # Sélection des colonnes finales pour l'entraînement
    features_columns = [
        'timestamp', 'close', 'return_5m', 'return_15m', 
        'volatility_60m', 'volume_surge', 'rsi_14', 'target'
    ]
    final_df = df[features_columns]
    
    final_df.to_csv(output_csv, index=False)
    print(f"✅ Fichier d'entraînement généré avec succès : {output_csv}")
    print(final_df.head())

if __name__ == "__main__":
    # Assure-toi d'avoir un fichier source valide et d'avoir fait un 'pip install ta'
    create_lgbm_features(r"Polymarket_BTC_5m\data\binance_1m_raw.csv", r"Polymarket_BTC_5m\data\final_features_v2.csv")