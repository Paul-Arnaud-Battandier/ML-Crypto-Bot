import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

def apply_triple_barrier(df, events, pt_vol_mult=1.0, sl_vol_mult=1.0, time_limit=6):
    """
    Applique la Triple Barrière de l'AFML.
    
    Paramètres :
    - df : Le dataframe complet avec les prix ('close') et la volatilité ('volatility_24h').
    - events : L'index (timestamps) des moments où notre XGBoost a déclenché un signal d'achat.
    - pt_vol_mult : Multiplicateur du Take Profit (Barrière Haute).
    - sl_vol_mult : Multiplicateur du Stop Loss (Barrière Basse).
    - time_limit : Nombre de bougies max avant fermeture (Barrière Verticale). 6 bougies de 4H = 24H.
    """
    labels = pd.Series(index=events, dtype=float)
    
    for t_event in events:
        # 1. On récupère les infos au moment exact du trade
        start_price = df.loc[t_event, 'close']
        volatility = df.loc[t_event, 'volatility_24h'] 
        
        # 2. Définition des barrières horizontales dynamiques (en prix absolu)
        take_profit = start_price + (start_price * volatility * pt_vol_mult)
        stop_loss = start_price - (start_price * volatility * sl_vol_mult)
        
        # 3. On regarde uniquement le futur jusqu'à la limite de temps
        # On extrait la fenêtre de prix du trade
        start_idx = df.index.get_loc(t_event)
        end_idx = min(start_idx + time_limit + 1, len(df))
        future_window = df.iloc[start_idx+1 : end_idx]
        
        label = 0 # Par défaut (SL ou Temps expiré)
        
        # 4. Simulation temporelle : qu'est-ce qui est touché en premier ?
        for current_time, row in future_window.iterrows():
            if row['low'] <= stop_loss:
                label = 0  # Stop Loss touché -> Échec
                break
            elif row['high'] >= take_profit:
                label = 1  # Take Profit touché -> Succès
                break
                
        labels[t_event] = label
        
    return labels



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
    # On dit au bot : "Ne trade que quand tu es sûr à plus de 50% que ça va monter"
    seuil_confiance = 0.52
    events = cv_probas[cv_probas > seuil_confiance].index
    print(f"\n🎯 Nombre de signaux d'achat détectés : {len(events)}")
    
    # 3. Le Crash-Test : La Triple Barrière
    print("🚧 Application de la Triple Barrière Dynamique...")
    # On met un Profit Taking à 1.5x la Volatilité, et un Stop Loss à 1x la Volatilité (Ratio 1.5:1)
    # Ratio 1:1 avec plus d'oxygène, et on laisse 48h au trade (12 bougies de 4H)
    meta_labels = apply_triple_barrier(df, events, pt_vol_mult=3.0, sl_vol_mult=1.5, time_limit=18)

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
    
    df_meta.to_csv("data/meta_dataset.csv")
    print("\n✅ Dataset du Meta-Modèle sauvegardé avec succès dans 'data/meta_dataset.csv'")