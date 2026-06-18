import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

def apply_fixed_horizon(df, events, horizon=2, fee_hurdle=0.002):
    """
    Cible Point-à-Point : Regarde uniquement le prix à t + horizon.
    Ignore totalement ce qui se passe entre les deux (Pas de Stop Loss, pas de Take Profit).
    
    - horizon : Nombre de bougies (2 bougies = 8 heures, donc de 16h à minuit)
    - fee_hurdle : La barrière des frais (0.002 = 0.2% pour couvrir l'aller-retour)
    """
    labels = pd.Series(index=events, dtype=float)
    stats = {'GAGNANT': 0, 'PERDANT': 0}
    
    for t_event in events:
        start_price = df.loc[t_event, 'close']
        start_idx = df.index.get_loc(t_event)
        
        # On vérifie qu'on a bien accès à la bougie de minuit
        if start_idx + horizon < len(df):
            final_price = df.iloc[start_idx + horizon]['close']
            
            # Pour gagner, le prix final doit battre le prix initial + les frais
            if final_price > start_price * (1 + fee_hurdle):
                labels[t_event] = 1.0
                stats['GAGNANT'] += 1
            else:
                labels[t_event] = 0.0
                stats['PERDANT'] += 1
        else:
            labels[t_event] = np.nan
            
    print("\n🔍 DÉTAIL DU FIXED HORIZON (16h00 -> 00h00) :")
    print(f"-> 🟢 Clôture en Gain Net (Frais payés) : {stats['GAGNANT']}")
    print(f"-> 🔴 Clôture en Perte                : {stats['PERDANT']}")
    print("-" * 40)
    
    return labels.dropna()


if __name__ == "__main__":
    print("⏳ Chargement des données...")
    df = pd.read_csv("data/final_features.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').set_index('timestamp')
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    # 1. Le Modèle Primaire avec TES hyperparamètres Optuna
    print("🤖 Simulation des prédictions (Out-of-Sample)...")
    model = xgb.XGBClassifier(
        n_estimators=120, max_depth=2, learning_rate=0.001289,
        subsample=0.85, colsample_bytree=0.92, min_child_weight=2,
        random_state=42, n_jobs=-1
    )
    
    # On génère les probabilités proprement sans tricher avec le futur
    tscv = TimeSeriesSplit(n_splits=5, gap=12)
    cv_probas = pd.Series(index=df.index, dtype=float)
    
    for train_idx, test_idx in tscv.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        # On stocke la probabilité de hausse (classe 1)
        preds = model.predict_proba(X.iloc[test_idx])[:, 1]
        cv_probas.iloc[test_idx] = preds
        
    # On enlève la première période (qui a servi d'entraînement initial et n'a pas de prédiction)
    cv_probas = cv_probas.dropna()
    
   # 2. Définition des "Trade Events"
    seuil_confiance = 0.50
    events_bruts = cv_probas[cv_probas > seuil_confiance].index
    print(f"\n🎯 Nombre de signaux d'achat initiaux : {len(events_bruts)}")
    
    # --- LE NOUVEAU FILTRE : SAISONNALITÉ INTRADAY ---
    print("⏰ Application du filtre Intraday (Entrée après 4 bougies)...")
    # On ne garde QUE les timestamps de 16h00
    events = events_bruts[events_bruts.hour == 16]
    print(f"🛡️ Signaux conservés pour le trade du soir : {len(events)}")
    # --------------------------------------------------
    
    # 3. Le Crash-Test : Fixed Time Horizon
    print("🚧 Application de la Cible Point-à-Point (00h00)...")
    # horizon=2 (pour avancer de deux bougies de 4h)
    meta_labels = apply_fixed_horizon(df, events, horizon=2, fee_hurdle=0.002)

    # 4. Analyse et Synthèse
    print("\n📊 Résultats des trades simulés :")
    print(meta_labels.value_counts().rename({1.0: "Succès (TP touché)", 0.0: "Échec (SL ou Temps expiré)"}))
    
    win_rate = meta_labels.mean() * 100
    print(f"-> Taux de réussite brut : {win_rate:.2f}%")
    
    # 5. Création du Dataset de la Phase 2 (Le Meta-Dataset)
    # On ne garde QUE les lignes où il y a eu un trade !
    df_meta = df.loc[events].copy()
    # On ajoute la confiance du premier modèle (très important pour le second)
    df_meta['primary_proba'] = cv_probas[events]
    # La nouvelle cible à prédire par le Random Forest : est-ce que ce trade a marché ?
    df_meta['meta_target'] = meta_labels.astype(int)
    
    # Nettoyage final des colonnes inutiles pour le Meta-Modèle
    df_meta = df_meta.drop(columns=['target']) 
    
    df_meta.index.name = 'timestamp'

    df_meta.to_csv("data/meta_dataset.csv")
    print("\n✅ Dataset du Meta-Modèle sauvegardé avec succès dans 'data/meta_dataset.csv'")