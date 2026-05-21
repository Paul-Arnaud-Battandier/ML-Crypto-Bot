import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

def apply_triple_barrier(df, events, pt_vol_mult=1.0, sl_vol_mult=1.0, time_limit=6):
    labels = pd.Series(index=events, dtype=float)
    
    # --- NOS NOUVEAUX COMPTEURS ---
    stats = {'TP': 0, 'SL': 0, 'TIME_POS': 0, 'TIME_NEG': 0}
    
    for t_event in events:
        start_price = df.loc[t_event, 'close']
        volatility = df.loc[t_event, 'volatility_24h'] 
        
        # Inversion pour le Short : TP en bas, SL en haut
        take_profit = start_price - (start_price * volatility * pt_vol_mult) 
        stop_loss = start_price + (start_price * volatility * sl_vol_mult)
        
        start_idx = df.index.get_loc(t_event)
        end_idx = min(start_idx + time_limit + 1, len(df))
        future_window = df.iloc[start_idx+1 : end_idx]
        
        touched = False
        
        for current_time, row in future_window.iterrows():
            # --- ATTENTION : TEST INVERSÉ (SHORT) ---
            if row['high'] >= stop_loss:  # Si le prix MONTE, on touche le Stop Loss (Perte)
                labels[t_event] = 0
                stats['SL'] += 1
                touched = True
                break
            elif row['low'] <= take_profit: # Si le prix BAISSE, on touche le Take Profit (Gain)
                labels[t_event] = 1
                stats['TP'] += 1
                touched = True
                break
                
        # Si la boucle se termine et que rien n'a été touché : Barrière Temps !
        if not touched:
            final_price = future_window.iloc[-1]['close'] if len(future_window) > 0 else start_price
            
            if final_price > start_price:
                # Expiré avec un petit profit ! (Mais on le laisse à 0 pour le ML 
                # car il n'a pas atteint notre vrai objectif de Take Profit)
                labels[t_event] = 0 
                stats['TIME_POS'] += 1
            else:
                # Expiré en perte
                labels[t_event] = 0
                stats['TIME_NEG'] += 1
                
    # --- AFFICHAGE DU RAPPORT D'AUTOPSIE ---
    print("\n🔍 DÉTAIL DES SORTIES DE TRADES :")
    print(f"-> 🟢 Touché Take Profit (TP)     : {stats['TP']}")
    print(f"-> 🔴 Touché Stop Loss (SL)       : {stats['SL']}")
    print(f"-> ⏱️ Expiré Temps (Gain partiel) : {stats['TIME_POS']}")
    print(f"-> ⏱️ Expiré Temps (Perte légère) : {stats['TIME_NEG']}")
    print("-" * 40)
    
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
    seuil_confiance = 0.50
    events_bruts = cv_probas[cv_probas > seuil_confiance].index
    print(f"\n🎯 Nombre de signaux d'achat initiaux : {len(events_bruts)}")
    
    # --- LE NOUVEAU FILTRE : SAISONNALITÉ INTRADAY ---
    print("⏰ Application du filtre Intraday (Entrée après 4 bougies)...")
    # On ne garde QUE les timestamps de 16h00
    events = events_bruts[events_bruts.hour == 16]
    print(f"🛡️ Signaux conservés pour le trade du soir : {len(events)}")
    # --------------------------------------------------
    
    # 3. Le Crash-Test : La Triple Barrière (Version Sprint 8H)
    print("🚧 Application de la Triple Barrière Dynamique (Barrières rapprochées)...")
    
    # Ancien TP = 1.5 / Ancien SL = 1.0 (Pour 24h)
    meta_labels = apply_triple_barrier(df, events, pt_vol_mult=0.75, sl_vol_mult=0.5, time_limit=2)

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