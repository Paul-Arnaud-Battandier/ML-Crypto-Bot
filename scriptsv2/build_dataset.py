import pandas as pd
from datav2 import fetch_binance_1m_data
from featuresv2 import create_features
from labelsv2 import create_polymarket_target

def build_final_dataset():
    print("🚀 DÉMARRAGE DE L'USINE À DONNÉES (Polymarket 5-Min MVP) 🚀\n")
    
    # 1. Extraction des données brutes
    # On demande ~10 000 minutes d'historique
    df_raw = fetch_binance_1m_data(symbol="BTCUSDT", loops=10)
    
    # 2. Création des Features (Les variables explicatives X)
    df_features = create_features(df_raw)
    
    # 3. Création de la Target (La variable à prédire y)
    # Attention: On crée la Target à partir des prix bruts, car df_features n'a plus la colonne 'close' !
    df_labeled = create_polymarket_target(df_raw, horizon=5)
    
    # 4. LA FUSION (Jointure)
    print("🔗 Fusion des Features et de la Target...")
    # On utilise concat sur l'index (timestamp). 
    # join='inner' garantit qu'on ne garde que les lignes où l'on a À LA FOIS les features ET la target (zéro NaN).
    df_final = pd.concat([df_features, df_labeled[['target']]], axis=1, join='inner')
    
    # 5. Bilan et Sauvegarde
    print("\n" + "="*40)
    print("✅ DATASET FINAL PRÊT POUR L'ENTRAÎNEMENT")
    print(f"-> Nombre total d'exemples (lignes) : {len(df_final)}")
    print(f"-> Nombre de features (colonnes X)  : {df_final.shape[1] - 1}")
    
    # Aperçu de l'équilibre des classes
    repartition = df_final['target'].value_counts(normalize=True) * 100
    print(f"-> Équilibre : Hausse {repartition.get(1, 0):.1f}% | Baisse/Neutre {repartition.get(0, 0):.1f}%")
    print("="*40 + "\n")
    
    # Sauvegarde
    output_path = "data/final_features_v2.csv"
    df_final.to_csv(output_path)
    print(f"💾 Sauvegardé avec succès dans : {output_path}")

if __name__ == "__main__":
    build_final_dataset()