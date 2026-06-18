import pandas as pd
import numpy as np
import os

def build_features(df):
    """
    Transforme les données brutes (OHLCV + Macro) en Feature Set.
    C'est ici que réside ton 'Alpha'.
    """
    # Copie pour éviter les SettingWithCopyWarning
    feat = df.copy()

    # --- 1. PRICE ACTION ---
    # Log Return (Base de calcul)
    feat['log_return'] = np.log(feat['close'] / feat['close'].shift(1))
    
    # Structure de la bougie
    feat['candle_body'] = (feat['close'] - feat['open']) / feat['open']
    feat['upper_wick'] = (feat['high'] - feat[['open', 'close']].max(axis=1)) / feat['open']
    feat['lower_wick'] = (feat[['open', 'close']].min(axis=1) - feat['low']) / feat['open']
    
    # Momentum Volume
    feat['volume_change'] = np.log(feat['volume'] / feat['volume'].shift(1))

    # --- 2. VOLATILITÉ ---
    # 6 bougies = 24h | 42 bougies = 7j
    feat['volatility_24h'] = feat['log_return'].rolling(window=6).std()
    feat['volatility_7d'] = feat['log_return'].rolling(window=42).std()
    
    # ATR (Average True Range) normalisé
    tr1 = feat['high'] - feat['low']
    tr2 = np.abs(feat['high'] - feat['close'].shift(1))
    tr3 = np.abs(feat['low'] - feat['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    feat['atr_14'] = true_range.rolling(window=14).mean() / feat['close']

    # --- 3. MOMENTUM ---
    # RSI 14
    delta = feat['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    feat['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Returns passés
    feat['return_24h'] = np.log(feat['close'] / feat['close'].shift(6))
    feat['return_7d'] = np.log(feat['close'] / feat['close'].shift(42))
    
    # Distance SMA 50
    feat['price_vs_sma50'] = (feat['close'] / feat['close'].rolling(window=50).mean()) - 1

    # --- 4. FUNDING RATE ---
    feat['funding_cumsum_3d'] = feat['funding_rate'].rolling(window=18).sum()

    # --- 5. MACRO (LEAD-LAG) ---
    nasdaq_ret = np.log(feat['nasdaq'] / feat['nasdaq'].shift(1))
    dxy_ret = np.log(feat['dxy'] / feat['dxy'].shift(1))
    
    feat['nasdaq_ret_lag4'] = nasdaq_ret.shift(4)
    feat['dxy_ret_lag1'] = dxy_ret.shift(1)

    # --- 6. TARGET (Modèle Primaire : Horizon 24h) ---
    # Question simple : le prix sera-t-il plus haut dans 24h (6 bougies de 4H) ?
    # On utilise le log return pour la propreté mathématique
    future_return = np.log(feat['close'].shift(-6) / feat['close'])
    feat['target'] = (future_return > 0).astype(int)

    # --- NETTOYAGE ---
    # On drop les NaNs créés par les rolling/lags.
    # IMPORTANT : Le shift(-6) va créer des NaNs sur les 6 DERNIÈRES lignes du dataset.
    # Le dropna() va automatiquement les supprimer pour éviter tout Data Leakage.
    feat = feat.dropna()

    print("Distribution du Target (24h) :")
    print(feat['target'].value_counts(normalize=True) * 100)
    
    return feat # ou selon comment ta fonction se termine

if __name__ == "__main__":
    # Chemins des fichiers (ajustés pour être lancés depuis la racine du projet)
    input_file = 'data/processed/master_data.csv'
    output_file = 'data/final_features.csv'

    # Vérification de l'existence du dossier data
    if not os.path.exists('data'):
        os.makedirs('data')

    print(f"--- Démarrage de la Features Factory ---")
    
    if os.path.exists(input_file):
        # Chargement des données
        master_df = pd.read_csv(input_file, index_col=0, parse_dates=True)
        print(f"Données brutes chargées : {len(master_df)} lignes.")

        # Génération des features
        final_df = build_features(master_df)
        print(f"Features générées : {final_df.shape[1]} colonnes.")

        # Sauvegarde
        final_df.to_csv(output_file)
        print(f"✅ Succès ! Fichier sauvegardé ici : {output_file}")
        print(f"Nombre final de lignes prêtes pour le ML : {len(final_df)}")
        
    else:
        print(f"❌ Erreur : Le fichier {input_file} est introuvable.")