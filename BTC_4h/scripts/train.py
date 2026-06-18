import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def run_duel(filepath):
    # 1. Chargement et tri chronologique
    print("⏳ Chargement des données...")
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').set_index('timestamp')
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    # 2. Validation croisée purgée (AFML style)
    # gap=12 bougies de 4H = 48h de "trou"
    tscv = TimeSeriesSplit(n_splits=5, gap=12)
    
    models = {
        "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
    }
    
    for name, model in models.items():
        print(f"\n{'='*10} Évaluation de {name} {'='*10}")
        auc_scores, acc_scores = [], []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            
            preds = model.predict(X_test)
            probas = model.predict_proba(X_test)[:, 1]
            
            auc_scores.append(roc_auc_score(y_test, probas))
            acc_scores.append(accuracy_score(y_test, preds))
            
        print(f"-> AUC Moyenne      : {np.mean(auc_scores):.4f}")
        print(f"-> Accuracy Moyenne : {np.mean(acc_scores):.4f}")
        
        print("\nRapport détaillé du dernier Fold :")
        print(classification_report(y_test, preds))

if __name__ == "__main__":
    # Assure-toi que ce chemin pointe bien vers ton fichier généré hier !
    run_duel("data/final_features.csv")