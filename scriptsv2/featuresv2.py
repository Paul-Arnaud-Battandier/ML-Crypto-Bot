import pandas as pd
import numpy as np

def create_features(df):
    """
    Transforme les prix bruts (1 minute) en indicateurs mathématiques nerveux.
    """
    print("🧠 Calcul des Features (Micro-structure & Momentum)...")
    df = df.copy()

    # 1. Rendements (Returns) purs
    df['return_1m'] = df['close'].pct_change(1)
    df['return_5m'] = df['close'].pct_change(5)
    
    # 2. Volatilité locale (sur 15 minutes)
    df['volatility_15m'] = df['return_1m'].rolling(window=15).std()
    
    # 3. Dynamique du Volume (Chocs d'activité)
    # Le volume brut ne veut rien dire, on veut savoir s'il y a un "pic" anormal
    df['volume_change'] = df['volume'].pct_change(1)
    df['volume_ma_15'] = df['volume'].rolling(window=15).mean()
    df['volume_surge'] = df['volume'] / df['volume_ma_15'] # Ratio: > 1 = pic de volume
    
    # 4. Micro-structure de la bougie (L'anatomie du prix)
    # On regarde la taille du corps et des mèches (en pourcentage du prix d'ouverture)
    df['candle_body'] = (df['close'] - df['open']) / df['open']
    
    # Mèche haute = Pression vendeuse cachée / Mèche basse = Pression acheteuse cachée
    df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
    df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
    
    # 5. Momentum : RSI (Relative Strength Index) fait maison sur 14 minutes
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # --- NETTOYAGE ---
    # Les calculs de fenêtres glissantes (rolling) créent des NaN au début du dataset.
    df = df.dropna()
    
    # On supprime les colonnes de prix bruts (optionnel mais recommandé en ML)
    # Les modèles ML préfèrent les données normalisées/stationnaires (pourcentages, ratios)
    cols_to_drop = ['open', 'high', 'low', 'close', 'volume', 'volume_ma_15']
    df_features = df.drop(columns=cols_to_drop)
    
    print(f"✅ Features créées avec succès : {df_features.shape[1]} variables générées.")
    return df_features

if __name__ == "__main__":
    # --- TEST RAPIDE ---
    print("⏳ Chargement des données brutes...")
    try:
        df_raw = pd.read_csv("data/binance_1m_raw.csv", index_col='timestamp', parse_dates=True)
        df_feats = create_features(df_raw)
        
        # Aperçu
        print("\nAperçu des nouvelles features :")
        print(df_feats.tail())
        
        # Sauvegarde
        df_feats.to_csv("data/features_1m.csv")
        print("\n💾 Features sauvegardées dans 'data/features_1m.csv'")
        
    except FileNotFoundError:
        print("❌ Erreur : Le fichier 'data/binance_1m_raw.csv' est introuvable. Lance datav2.py en premier !")