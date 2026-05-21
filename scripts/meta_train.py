import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

def train_meta_model():
    print("⏳ Chargement du Meta-Dataset...")
    df = pd.read_csv("data/meta_dataset.csv", index_col='timestamp', parse_dates=True)
    
    X = df.drop(columns=['meta_target'])
    y = df['meta_target']
    
    print(f"Taille du dataset : {len(df)} trades à analyser.")
    
    tscv = TimeSeriesSplit(n_splits=5, gap=12)
    meta_model = RandomForestClassifier(
        n_estimators=200, max_depth=5, class_weight='balanced', random_state=42, n_jobs=-1
    )
    
    thresholds = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
    
    # Nouveau dictionnaire pour accumuler les VRAIS chiffres globaux
    results = {t: {'total_trades': 0, 'vrais_gagnants': 0} for t in thresholds}
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        meta_model.fit(X_train, y_train)
        meta_probas = meta_model.predict_proba(X_test)[:, 1]
        
        for t in thresholds:
            custom_preds = (meta_probas > t).astype(int)
            
            # On compte mathématiquement les gagnants et les trades pris sur CE fold
            trades_pris = sum(custom_preds == 1)
            gagnants = sum((custom_preds == 1) & (y_test == 1))
            
            # On les ajoute au grand total
            results[t]['total_trades'] += trades_pris
            results[t]['vrais_gagnants'] += gagnants

    print("\n📊 --- ANALYSE GLOBALE DES SEUILS (VRAIE PRÉCISION) ---")
    print("Rappel théorique : > 40.0% pour Break-even SANS frais (Ratio 1.5:1).")
    print("Objectif réel    : > 45.0% pour être RENTABLE AVEC frais.\n")
    
    for t in thresholds:
        total_trades = results[t]['total_trades']
        gagnants = results[t]['vrais_gagnants']
        
        if total_trades > 0:
            precision_reelle = (gagnants / total_trades) * 100
        else:
            precision_reelle = 0.0
            
        # L'analyse implacable corrigée (Ratio 1.5:1)
        if precision_reelle > 45.0:
            etat = "✅ RENTABLE (Survit aux frais)"
        elif precision_reelle > 40.0:
            etat = "⚠️ BREAK-EVEN (Tué par les frais)"
        else:
            etat = "❌ PERTE"
            
        print(f"Seuil {t*100:.0f}% -> Vraie Précision: {precision_reelle:.2f}% | Trades totaux: {total_trades} | {etat}")
        
if __name__ == "__main__":
    train_meta_model()