import pandas as pd
import numpy as np

def create_polymarket_target(df, horizon=5):
    """
    Crée la Target pour le marché Polymarket 5-min.
    Horizon = 5 bougies de 1 minute.
    Target = 1 si le prix à t+5 est strictement supérieur au prix actuel (Hausse).
    Target = 0 sinon (Baisse ou Stagnation).
    """
    print(f"⏳ Création de la cible (Horizon : +{horizon} minutes)...")
    
    # 1. On va chercher le prix de clôture dans le futur
    df['future_close'] = df['close'].shift(-horizon)
    
    # 2. La règle binaire (1 = UP, 0 = DOWN)
    df['target'] = np.where(df['future_close'] > df['close'], 1, 0)
    
    # 3. Purge du futur : Les 5 dernières lignes n'ont pas de t+5, on les supprime.
    df = df.dropna(subset=['future_close'])
    
    # 4. Anti-Leakage : On supprime la colonne du futur !
    df = df.drop(columns=['future_close'])
    
    # Affichage des statistiques
    repartition = df['target'].value_counts(normalize=True) * 100
    print("\n📊 Répartition des Labels :")
    print(f"-> 🟢 Hausse (1) : {repartition.get(1, 0):.2f}%")
    print(f"-> 🔴 Baisse (0) : {repartition.get(0, 0):.2f}%")
    print("-" * 40)
    
    return df

if __name__ == "__main__":
    # Test unitaire rapide
    print("⏳ Chargement des données brutes de Binance (1 minute)...")
    # Simulation avec tes données (il faudra utiliser le data.py de Binance)
    # df = pd.read_csv("data/binance_1m_raw.csv")
    
    # df_labeled = create_polymarket_target(df)
    # df_labeled.to_csv("data/labeled_features.csv", index=False)
    # print("✅ Fichier sauvegardé !")
    pass