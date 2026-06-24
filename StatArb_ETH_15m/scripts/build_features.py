import pandas as pd
import numpy as np
from pathlib import Path

# --- Chemins ---
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
INPUT_FILE = DATA_DIR / "historical_AAVE_ETH_15m.csv"
OUTPUT_FILE = DATA_DIR / "ml_dataset.csv"

def main():
    print("="*60)
    print("🧠 ÉTAPE 2 : DATA ENGINEERING & META-LABELING")
    print("="*60)
    
    print("1️⃣ Chargement des données historiques...")
    df = pd.read_csv(INPUT_FILE, index_col='timestamp', parse_dates=True)
    
    # ---------------------------------------------------------
    # A. CALCUL DU Z-SCORE GLISSANT (Le Moteur Statistique)
    # ---------------------------------------------------------
    print("2️⃣ Calcul du modèle mathématique (Z-Score Rolling = 200 bougies)...")
    window = 200
    # --- LE VRAI HEDGE RATIO INSTITUTIONNEL (Rolling OLS Vectorisé) ---
    # Covariance glissante entre AAVE et ETH
    rolling_cov = df['AAVE'].rolling(window).cov(df['ETH'])
    # Variance glissante de ETH
    rolling_var = df['ETH'].rolling(window).var()

    # Le Beta (La pente de la régression linéaire) devient notre Ratio dynamique
    df['ratio'] = rolling_cov / rolling_var

    # Le nouveau Spread parfaitement couvert
    df['spread'] = df['AAVE'] - (df['ratio'] * df['ETH'])
    df['spread_mean'] = df['spread'].rolling(window).mean()
    df['spread_std'] = df['spread'].rolling(window).std()
    df['zscore'] = (df['spread'] - df['spread_mean']) / df['spread_std']
    
    # ---------------------------------------------------------
    # B. CRÉATION DES FEATURES ML (Le Contexte du Marché)
    # ---------------------------------------------------------
    print("3️⃣ Génération des indicateurs de contexte (Features)...")
    # Vitesse à laquelle le spread s'écarte (Momentum)
    df['zscore_mom_1h'] = df['zscore'].diff(4)
    df['zscore_mom_4h'] = df['zscore'].diff(16)
    
    # Rendements récents
    df['aave_ret_1h'] = df['AAVE'].pct_change(4)
    df['eth_ret_1h'] = df['ETH'].pct_change(4)
    
    # Volatilité locale (Écart-type des rendements)
    df['aave_vol_4h'] = df['AAVE'].pct_change().rolling(16).std()
    df['eth_vol_4h'] = df['ETH'].pct_change().rolling(16).std()
    
    # Nettoyage des NaN dus aux fenêtres glissantes
    df = df.dropna().copy()

    # ---------------------------------------------------------
    # C. TRIPLE BARRIER METHOD (Création des Labels 0/1)
    # ---------------------------------------------------------
    print("4️⃣ Simulation des trades passés (Triple Barrier Method)...")
    signals = []
    in_position = False
    
    # Optimisation des accès (Numpy arrays pour aller très vite)
    z_values = df['zscore'].values
    timestamps = df.index
    
    # Limite de temps pour un trade : 24 heures (96 bougies de 15m)
    TIME_LIMIT = 96 
    
    for i in range(len(df) - TIME_LIMIT):
        z = z_values[i]
        
        # Si on n'est pas déjà dans un trade et qu'on touche les extrêmes
        if not in_position:
            if z >= 2.0 or z <= -2.0:
                in_position = True
                entry_z = z
                direction = 1 if z <= -2.0 else -1  # 1 = Long Spread, -1 = Short Spread
                
                label = 0 # Par défaut, on part du principe que c'est un échec
                
                # Regardons dans le futur pour voir ce qui s'est passé...
                for j in range(i+1, i + TIME_LIMIT):
                    fut_z = z_values[j]
                    
                    # BARRIÈRE 1 : TAKE PROFIT (Retour à la moyenne parfait)
                    if (direction == 1 and fut_z >= 0) or (direction == -1 and fut_z <= 0):
                        label = 1
                        in_position = False
                        break
                        
                    # BARRIÈRE 2 : STOP LOSS (Cygne noir, Z-Score explose à 4)
                    if (direction == 1 and fut_z <= -4.0) or (direction == -1 and fut_z >= 4.0):
                        label = 0
                        in_position = False
                        break
                        
                # BARRIÈRE 3 : TIME LIMIT (Si la boucle finit sans toucher le TP ou le SL)
                if in_position:
                    label = 0  # Le trade a mis trop de temps, c'est un échec
                    in_position = False
                
                # Sauvegarde de ce scénario d'entraînement pour le ML
                signals.append({
                    'timestamp': timestamps[i],
                    'zscore_entry': entry_z,
                    'zscore_mom_1h': df['zscore_mom_1h'].iloc[i],
                    'zscore_mom_4h': df['zscore_mom_4h'].iloc[i],
                    'aave_ret_1h': df['aave_ret_1h'].iloc[i],
                    'eth_ret_1h': df['eth_ret_1h'].iloc[i],
                    'aave_vol_4h': df['aave_vol_4h'].iloc[i],
                    'eth_vol_4h': df['eth_vol_4h'].iloc[i],
                    'label': label
                })

    # ---------------------------------------------------------
    # D. EXPORT DU DATASET ML
    # ---------------------------------------------------------
    ml_df = pd.DataFrame(signals)
    ml_df.set_index('timestamp', inplace=True)
    
    print("\n" + "="*60)
    print(f"✅ DATASET TERMINÉ : {len(ml_df)} signaux détectés en 6 mois.")
    print(f"📊 Ratio de réussite brut (Moteur Stat) : {ml_df['label'].mean()*100:.2f}% (Objectif du ML : Améliorer ça)")
    
    ml_df.to_csv(OUTPUT_FILE)
    print(f"💾 Fichier d'entraînement sauvegardé : {OUTPUT_FILE}")
    print("="*60)

if __name__ == "__main__":
    main()