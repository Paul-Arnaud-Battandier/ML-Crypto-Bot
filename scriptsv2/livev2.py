import time
import datetime
import lightgbm as lgb
import pandas as pd
from scriptsv2.datav2 import fetch_binance_1m_data
from scriptsv2.featuresv2 import create_features
from scriptsv2.polymarket_data import get_live_btc_5m_market
import os
import csv
import warnings
warnings.filterwarnings('ignore')

def train_live_oracle():
    """Entraîne le modèle LightGBM en 2 secondes avec les données récentes."""
    print("🧠 Entraînement de l'Oracle en cours...")
    try:
        df = pd.read_csv("data/final_features_v2.csv", index_col='timestamp', parse_dates=True)
        X = df.drop(columns=['target'])
        y = df['target']
        
        model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05, 
            random_state=42, n_jobs=-1, verbose=-1
        )
        model.fit(X, y)
        print("✅ Oracle prêt et entraîné !")
        return model, X.columns
    except FileNotFoundError:
        print("❌ Erreur : Fichier 'final_features_v2.csv' manquant. Lance build_dataset.py d'abord.")
        return None, None

def get_current_oracle_prediction(model, feature_columns):
    """Télécharge les 20 dernières minutes de Binance et génère la prédiction."""
    try:
        # On télécharge juste 1 boucle (1000 minutes) pour être rapide
        df_raw = fetch_binance_1m_data(loops=1)
        df_feats = create_features(df_raw)
        
        # On prend la toute dernière ligne (la minute actuelle)
        latest_features = df_feats.iloc[-1:]
        
        # Sécurité : on s'assure d'avoir les bonnes colonnes dans le bon ordre
        latest_features = latest_features[feature_columns]
        
        # Probabilité de Hausse (Classe 1)
        proba_up = model.predict_proba(latest_features)[0][1]
        return proba_up
        
    except Exception as e:
        print(f"❌ Erreur lors du calcul de la prédiction : {e}")
        return None

def log_trade_to_csv(timestamp, market_name, oracle_prob, poly_price, edge, decision):
    """Enregistre le trade virtuel dans un fichier CSV pour le dashboard futur."""
    file_path = "data/trade_history.csv"
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Création de l'en-tête si le fichier est nouveau
        if not file_exists:
            writer.writerow(["timestamp", "market_name", "oracle_prob", "poly_price", "edge", "decision"])
        writer.writerow([timestamp, market_name, oracle_prob, poly_price, edge, decision])

def run_trading_bot():
    """La Boucle Principale du Bot avec persistance des données"""
    print("\n" + "="*50)
    print("🚀 DÉMARRAGE DU TRADING BOT (VERSION CLOUD-READY) 🚀")
    print("="*50 + "\n")
    
    model, feature_columns = train_live_oracle()
    if not model: return
    
    print("\n⏳ En attente de la prochaine minute ronde...")
    
    while True:
        now = datetime.datetime.now()
        
        if now.second == 5:
            print(f"\n⏰ {now.strftime('%H:%M:%S')} - Analyse en cours...")
            
            proba_up = get_current_oracle_prediction(model, feature_columns)
            market_data = get_live_btc_5m_market()
            
            if proba_up is not None and market_data and market_data['ask_price'] is not None:
                oracle_prob_pct = round(proba_up * 100, 2)
                poly_price_pct = round(market_data['ask_price'] * 100, 2)
                ecart = round(oracle_prob_pct - poly_price_pct, 2)
                
                decision = "NEUTRAL"
                if ecart > 5.0: decision = "BUY_YES"
                elif ecart < -5.0: decision = "BUY_NO"
                
                print(f"-> ML: {oracle_prob_pct}% | Polymarket: {poly_price_pct}% | Edge: {ecart}% | Action: {decision}")
                
                # --- SAUVEGARDE DU TRADE ---
                if decision != "NEUTRAL":
                    log_trade_to_csv(
                        timestamp=now.strftime('%Y-%m-%d %H:%M:%S'),
                        market_name=market_data['market_name'],
                        oracle_prob=oracle_prob_pct,
                        poly_price=poly_price_pct,
                        edge=ecart,
                        decision=decision
                    )
                    print("💾 Trade virtuel sauvegardé dans l'historique !")
            
            time.sleep(50)
        time.sleep(0.5)

if __name__ == "__main__":
    run_trading_bot()