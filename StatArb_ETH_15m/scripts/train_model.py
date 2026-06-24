import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report
import joblib
from pathlib import Path

# --- Chemins ---
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

INPUT_FILE = DATA_DIR / "ml_dataset.csv"
MODEL_FILE = MODEL_DIR / "lgbm_statarb.pkl"

def main():
    print("="*60)
    print("🤖 ÉTAPE 3 : ENTRAÎNEMENT DU MODÈLE (LIGHTGBM)")
    print("="*60)

    # 1. Chargement des données
    try:
        df = pd.read_csv(INPUT_FILE, index_col='timestamp', parse_dates=True)
    except FileNotFoundError:
        print(f"❌ Erreur : Fichier introuvable à {INPUT_FILE}")
        return
    
    # 2. Séparation Features (X) / Target (y)
    # Ce sont les indicateurs que le ML va analyser pour prendre sa décision
    features = [
        'zscore_entry', 'zscore_mom_1h', 'zscore_mom_4h', 
        'aave_ret_1h', 'eth_ret_1h', 'aave_vol_4h', 'eth_vol_4h'
    ]
    
    X = df[features]
    y = df['label']
    
    # 3. Train/Test Split (Chronologique)
    # En finance, on ne mélange JAMAIS le passé et le futur (Shuffle=False)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"📊 Données d'entraînement : {len(X_train)} signaux (Le Passé)")
    print(f"📊 Données de test        : {len(X_test)} signaux (Le Futur out-of-sample)\n")
    
    # 4. Entraînement du LightGBM
    print("⚙️ Entraînement du modèle LightGBM...")
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        class_weight='balanced', # Force le modèle à donner autant d'importance aux pertes qu'aux gains
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    
    # 5. Évaluation avec Seuil de Confiance (Le cœur de notre Edge)
    # On obtient des probabilités de réussite (entre 0 et 1)
    proba = model.predict_proba(X_test)[:, 1] 
    
    threshold = 0.60 # Le ML doit être sûr à 60% minimum pour valider le trade
    high_conf_trades = proba > threshold
    
    total_test_trades = len(y_test)
    trades_taken = high_conf_trades.sum()
    
    print("\n" + "="*60)
    print(f"🎯 RÉSULTATS HAUTE CONFIANCE (Seuil ML > {threshold*100}%) :")
    print("="*60)
    
    if trades_taken > 0:
        precision_high_conf = y_test[high_conf_trades].mean()
        print(f"   - Opportunités proposées par le Z-Score : {total_test_trades}")
        print(f"   - Trades validés par le vigile ML       : {trades_taken} ({(trades_taken/total_test_trades)*100:.1f}%)")
        print(f"   - NOUVEAU WIN-RATE (Précision)          : {precision_high_conf*100:.2f}%")
        
        # Calcul de l'amélioration
        amelioration = precision_high_conf*100 - (y_test.mean()*100)
        color = "🟢" if amelioration > 0 else "🔴"
        print(f"   - Amélioration vs Moteur Brut           : {color} {amelioration:+.2f}%")
    else:
        print(f"   Aucun trade ne dépasse {threshold*100}% de confiance. Baisse le seuil.")

    # 6. Qu'est-ce qui a influencé le modèle ?
    print("\n🔍 IMPORTANCE DES FEATURES (Qu'est-ce que le ML regarde ?) :")
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    for index, row in importance.iterrows():
        print(f"   - {row['Feature']:<15} : {row['Importance']}")

    # 7. Sauvegarde
    joblib.dump(model, MODEL_FILE)
    print("\n" + "="*60)
    print(f"💾 Modèle ML sauvegardé : {MODEL_FILE}")
    print("="*60)

if __name__ == "__main__":
    main()