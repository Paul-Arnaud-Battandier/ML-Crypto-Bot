import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

def run_backtest():
    print("⏳ Chargement du Meta-Dataset et simulation des probabilités...")
    df = pd.read_csv("data/meta_dataset.csv", index_col='timestamp', parse_dates=True)
    
    X = df.drop(columns=['meta_target'])
    y = df['meta_target']
    
    # 1. On génère les probabilités du Meta-Modèle proprement (sans tricher)
    tscv = TimeSeriesSplit(n_splits=5, gap=12)
    meta_model = RandomForestClassifier(n_estimators=200, max_depth=5, class_weight='balanced', random_state=42, n_jobs=-1)
    
    df['meta_proba'] = np.nan
    for train_idx, test_idx in tscv.split(X):
        meta_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        df.iloc[test_idx, df.columns.get_loc('meta_proba')] = meta_model.predict_proba(X.iloc[test_idx])[:, 1]
        
    # On enlève les données d'entraînement initiales qui n'ont pas de prédictions
    df = df.dropna(subset=['meta_proba'])

    # ==========================================
    # 2. LA SIMULATION FINANCIÈRE (BACKTEST)
    # ==========================================
    print("💸 Démarrage de la simulation financière...")
    
    capital_initial = 10000.0
    capital = capital_initial
    frais_binance = 0.001  # 0.1% par ordre (pour couvrir frais Spot/Futures + Slippage)
    
    # Historique pour les statistiques
    historique_capital = [capital_initial]
    trades_pris = 0
    trades_gagnants = 0
    
    for idx, row in df.iterrows():
        proba = row['meta_proba']
        volatilite = row['volatility_24h']  # Ex: 0.03 pour 3% de volatilité ce jour-là
        
        # --- POSITION SIZING ---
        if proba >= 0.58:
            risque_pct = 0.05  # Trade Gros : 5% du capital
        elif proba >= 0.54:
            risque_pct = 0.02  # Trade Petit : 2% du capital
        else:
            risque_pct = 0.00  # On ignore
            
        # --- EXECUTION DU TRADE ---
        if risque_pct > 0:
            trades_pris += 1
            taille_position = capital * risque_pct
            
            # On paie les frais à l'entrée ET à la sortie
            cout_frais = taille_position * frais_binance * 2
            
            # Résultat du trade (grâce au meta_target calculé dans labels.py)
            if row['meta_target'] == 1:
                # Succès ! Le Take Profit (1.5x la volatilité) a été touché
                profit_brut = taille_position * (volatilite * 1.5)
                capital += (profit_brut - cout_frais)
                trades_gagnants += 1
            else:
                # Échec. Le Stop Loss (1.0x la volatilité) a été touché
                perte_brute = taille_position * (volatilite * 1.0)
                capital -= (perte_brute + cout_frais)
                
        historique_capital.append(capital)
        
    # ==========================================
    # 3. RÉSULTATS
    # ==========================================
    roi = ((capital - capital_initial) / capital_initial) * 100
    win_rate = (trades_gagnants / trades_pris) * 100 if trades_pris > 0 else 0
    
    print("\n" + "="*30)
    print("🏆 RÉSULTATS DU BACKTEST 🏆")
    print("="*30)
    print(f"Capital de départ : {capital_initial:.2f} USDT")
    print(f"Capital final     : {capital:.2f} USDT")
    print(f"Bénéfice Net      : {capital - capital_initial:.2f} USDT")
    print(f"ROI (Rendement)   : {roi:.2f}%")
    print("-" * 30)
    print(f"Nombre de trades  : {trades_pris}")
    print(f"Win Rate          : {win_rate:.2f}%")

if __name__ == "__main__":
    run_backtest()