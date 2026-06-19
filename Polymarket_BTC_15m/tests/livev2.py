import requests
import pandas as pd
import numpy as np
import ta
import json
import joblib
from datetime import datetime, timezone

# ==========================================
# 1. PARAMÈTRES ET CHEMINS
# ==========================================
MODEL_PATH = r"Polymarket_BTC_5m\model\lgbm_model.pkl"
# Seuil de sécurité pour le Spread (Si l'écart Bid/Ask est > 0.05$, on ne trade pas)
MAX_SPREAD_ALLOWED = 0.05  

# ==========================================
# 2. FONCTIONS DE RÉCUPÉRATION DE DONNÉES
# ==========================================
def get_live_binance_data():
    """Télécharge les 250 dernières minutes du BTC pour calculer les features"""
    print("📡 1. Récupération des prix BTC en direct (Binance)...")
    url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=250"
    res = requests.get(url)
    data = res.json()
    
    # On crée le DataFrame avec les mêmes colonnes que ton fichier d'entraînement
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    return df

def calculate_live_features(df):
    """Calcule exactement les mêmes indicateurs que featuresv2.py"""
    print("⚙️ 2. Calcul des indicateurs mathématiques (Mémoire du marché)...")
    df['return_5m'] = df['close'].pct_change(periods=5)
    df['return_15m'] = df['close'].pct_change(periods=15)
    df['volatility_60m'] = df['close'].pct_change().rolling(window=60).std()
    
    rolling_vol_15m = df['volume'].rolling(window=15).sum()
    rolling_vol_4h = df['volume'].rolling(window=240).mean() * 15
    df['volume_surge'] = rolling_vol_15m / rolling_vol_4h
    
    df['rsi_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    
    # On isole la TOUTE DERNIÈRE ligne (la minute actuelle)
    current_state = df.iloc[-1:].copy()
    features = ['return_5m', 'return_15m', 'volatility_60m', 'volume_surge', 'rsi_14']
    
    return current_state[features]

# ==========================================
# 3. FONCTIONS POLYMARKET (Le Sniper 15m)
# ==========================================
def get_polymarket_live_status():
    """Snipe le marché actuel et retourne le spread et les prix"""
    print("🎯 4. Interrogation de la liquidité Polymarket (Sniper 15m)...")
    now = datetime.now(timezone.utc)
    minute_start = (now.minute // 15) * 15
    current_15m = now.replace(minute=minute_start, second=0, microsecond=0)
    timestamp = int(current_15m.timestamp())
    slug = f"btc-updown-15m-{timestamp}"
    
    url_gamma = f"https://gamma-api.polymarket.com/events?slug={slug}"
    try:
        events = requests.get(url_gamma).json()
        if not events:
            return None, "Marché introuvable sur Gamma"
        
        token_yes = json.loads(events[0]['markets'][0]['clobTokenIds'])[0]
        book = requests.get(f"https://clob.polymarket.com/book?token_id={token_yes}").json()
        
        bids = book.get('bids', [])
        asks = book.get('asks', [])
        
        if bids and asks:
            best_bid = float(bids[0].get('price'))
            best_ask = float(asks[0].get('price'))
            return {"bid": best_bid, "ask": best_ask, "spread": best_ask - best_bid}, "OK"
        return None, "Carnet d'ordres vide (Ghost Town)"
    except Exception as e:
        return None, f"Erreur API: {e}"

# ==========================================
# 4. LE CERVEAU (Exécution du Bot)
# ==========================================
def run_bot():
    print("="*50)
    print(f"🚀 DÉMARRAGE DU BOT QUANTITATIF V1 - {datetime.now().strftime('%H:%M:%S')}")
    print("="*50)
    
    # Étape 1 & 2 : Data & Features
    raw_df = get_live_binance_data()
    live_features = calculate_live_features(raw_df)
    
    # Étape 3 : IA (Prédiction)
    print("🧠 3. Réveil de l'Intelligence Artificielle (LGBM)...")
    model = joblib.load(MODEL_PATH)
    
    prediction = model.predict(live_features)[0]
    probabilities = model.predict_proba(live_features)[0]
    
    direction = "UP (1)" if prediction == 1 else "DOWN (0)"
    confidence = probabilities[1] if prediction == 1 else probabilities[0]
    
    print(f"   -> DÉCISION LGBM : {direction} (Confiance : {confidence*100:.1f}%)")
    
    # Étape 4 : Le Filtre de Risque Polymarket (Proto Meta-Labeling)
    poly_data, status = get_polymarket_live_status()
    
    if not poly_data:
        print(f"❌ TRADE ANNULÉ : {status}")
        return

    print(f"   -> Spread actuel : {poly_data['spread']:.4f} $")
    print(f"   -> Prix d'Achat 'YES' (Ask) : {poly_data['ask']:.2f} $")
    print(f"   -> Prix de Vente 'YES' (Bid) : {poly_data['bid']:.2f} $")
    
    # Étape 5 : Exécution Logique
    print("\n⚖️ DÉCISION FINALE DU SYSTÈME :")
    if poly_data['spread'] > MAX_SPREAD_ALLOWED:
        print(f"🛑 PAS DE TRADE. Le spread ({poly_data['spread']:.2f}$) est supérieur à la limite de sécurité ({MAX_SPREAD_ALLOWED}$). Risque de friction trop élevé.")
    else:
        if prediction == 1:
            print(f"✅ ACHAT VALIDÉ : Le bot achèterait des parts 'YES' au prix de {poly_data['ask']:.2f} $.")
        else:
            print(f"✅ ACHAT VALIDÉ : Le bot achèterait des parts 'NO' (ou vendrait YES à {poly_data['bid']:.2f} $).")
            
    print("="*50)

if __name__ == "__main__":
    run_bot()