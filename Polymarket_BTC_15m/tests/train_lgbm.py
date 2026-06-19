import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_model(data_path, model_path):
    print(f"📖 Lecture des données d'entraînement depuis {data_path}...")
    df = pd.read_csv(data_path)
    
    # 1. Sélection des Features (On exclut explicitement 'timestamp', 'close' et 'target')
    features = [col for col in df.columns if col not in ['timestamp', 'close', 'target', 'future_close_15m']]
    
    print(f"🔍 Variables utilisées par l'IA : {features}")
    
    X = df[features]
    y = df['target']
    
    # 2. Séparation Temporelle (Time-Series Split)
    # Règle d'or en finance : on n'utilise JAMAIS de random split. 
    # On entraîne sur le passé (80%), on teste sur le futur (20%).
    split_index = int(len(df) * 0.80)
    
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    print(f"🏋️ Lignes d'entraînement : {len(X_train)} | Lignes de test : {len(X_test)}")
    
    # 3. Création et Entraînement du Modèle LightGBM
    print("\n⚙️ Entraînement du Modèle LGBM en cours...")
    model = lgb.LGBMClassifier(
        n_estimators=200,      # Nombre d'arbres
        learning_rate=0.05,    # Vitesse d'apprentissage
        max_depth=5,           # Profondeur des arbres (5 = évite le sur-apprentissage)
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # 4. Évaluation des performances sur les données qu'il n'a jamais vues
    print("\n📊 Évaluation des Performances (Données de Test) :")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"🎯 Précision Globale (Accuracy) : {accuracy * 100:.2f} %")
    print("\nRapport détaillé :")
    print(classification_report(y_test, y_pred, target_names=["DOWN (0)", "UP (1)"]))
    
    # 5. Sauvegarde du Cerveau
    joblib.dump(model, model_path)
    print(f"💾 Modèle sauvegardé avec succès sous : {model_path}")

if __name__ == "__main__":
    # Adapte les chemins selon ton dossier local
    DATA_FILE = r"Polymarket_BTC_15m\data\final_features_v2.csv"
    MODEL_FILE = r"Polymarket_BTC_15m\model\lgbm_model.pkl"
    
    # Création du dossier 'model' s'il n'existe pas
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    
    train_model(DATA_FILE, MODEL_FILE)