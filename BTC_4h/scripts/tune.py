import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# 1. Chargement global pour économiser la mémoire
print("⏳ Chargement des données...")
df = pd.read_csv("data/final_features.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').set_index('timestamp')
X = df.drop(columns=['target'])
y = df['target']

# 2. La fonction objectif d'Optuna
def objective(trial):
    # L'espace de recherche : Optuna va tester des milliers de combinaisons intelligemment
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 400),
        'max_depth': trial.suggest_int('max_depth', 2, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,
        'n_jobs': -1
    }
    
    tscv = TimeSeriesSplit(n_splits=5, gap=12)
    auc_scores = []
    
    for step, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = xgb.XGBClassifier(**param)
        model.fit(X_train, y_train)
        
        probas = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probas)
        auc_scores.append(auc)
        
        # --- L'Élagage (Pruning) ---
        # Si dès le 2ème fold le score est nul, Optuna abandonne ces paramètres
        trial.report(auc, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    return np.mean(auc_scores)

if __name__ == "__main__":
    print("🚀 Lancement d'Optuna (Durée max : 10 minutes)...")
    
    # Création de l'étude avec un algorithme de Pruning
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    
    # On lance l'optimisation (Max 50 essais OU 10 minutes)
    study.optimize(objective, n_trials=50, timeout=600)
    
    print("\n" + "="*30)
    print("🏆 OPTIMISATION TERMINÉE 🏆")
    print("="*30)
    print(f"Meilleure AUC moyenne : {study.best_value:.4f} (Ancien record : 0.5374)")
    print("\nCopie-colle ces hyperparamètres pour la suite :")
    for key, value in study.best_params.items():
        print(f"    '{key}': {value},")