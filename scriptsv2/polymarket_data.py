import requests
import time
import json

def get_live_btc_5m_market():
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }
    
    # 1. Prédiction Mathématique du Slug
    now = int(time.time())
    base_time = now - (now % 300)
    possible_timestamps = [base_time, base_time + 300, base_time + 600]
    
    target_market = None
    target_slug = ""
    
    for ts in possible_timestamps:
        slug = f"btc-updown-5m-{ts}"
        event_url = f"https://gamma-api.polymarket.com/events?slug={slug}"
        
        try:
            response = requests.get(event_url, headers=headers, timeout=5)
            if response.status_code == 200:
                events = response.json()
                if events and len(events) > 0:
                    markets = events[0].get('markets', [])
                    if markets and not markets[0].get('closed'):
                        target_market = markets[0]
                        target_slug = slug
                        break
        except Exception:
            continue
            
    if not target_market:
        print("⏳ Aucun marché ouvert trouvé.")
        return None
        
    print(f"\n🎯 Cible verrouillée mathématiquement : {target_slug}")
    
    # --- LA CORRECTION EST ICI ---
    # On récupère les données brutes
    outcomes_raw = target_market.get('outcomes', [])
    token_ids_raw = target_market.get('clobTokenIds', [])
    
    # Si Polymarket nous envoie une phrase au lieu d'une liste, on la décode
    if isinstance(outcomes_raw, str):
        try: outcomes = json.loads(outcomes_raw)
        except: outcomes = []
    else:
        outcomes = outcomes_raw
        
    if isinstance(token_ids_raw, str):
        try: token_ids = json.loads(token_ids_raw)
        except: token_ids = []
    else:
        token_ids = token_ids_raw
        
    # -----------------------------
    
    yes_token_id = None
    
    # On cherche "Up" ou "Yes"
    for i, outcome in enumerate(outcomes):
        if outcome.upper() in ['YES', 'UP']:
            if i < len(token_ids):
                yes_token_id = token_ids[i]
            break
            
    # Fallback
    if not yes_token_id and token_ids:
        yes_token_id = token_ids[0]
        
    if not yes_token_id:
        print("❌ Impossible d'isoler l'ID du token.")
        return None
        
    question = target_market.get('question', 'Marché BTC 5m')
    
    # 3. Requête du Carnet d'Ordres (CLOB)
    clob_url = f"https://clob.polymarket.com/book?token_id={yes_token_id}"
    try:
        clob_resp = requests.get(clob_url, headers=headers, timeout=10)
        clob_resp.raise_for_status()
        book_data = clob_resp.json()
        
        asks = book_data.get('asks', [])
        bids = book_data.get('bids', [])
        
        best_ask = float(asks[0]['price']) if asks else None
        best_bid = float(bids[0]['price']) if bids else None
        
        return {
            "market_name": question,
            "token_id": yes_token_id,
            "ask_price": best_ask,
            "bid_price": best_bid
        }
    except Exception as e:
        print(f"❌ Erreur lors de la lecture du carnet d'ordres : {e}")
        return None

if __name__ == "__main__":
    print("📡 Recherche du contrat BTC 5m en cours...")
    market_data = get_live_btc_5m_market()
    
    if market_data:
        print(f"✅ Marché : {market_data['market_name']}")
        print(f"🛒 Prix pour ACHETER le YES/UP : {market_data['ask_price']}$")
        print(f"💰 Prix pour VENDRE  le YES/UP : {market_data['bid_price']}$")
        if market_data['ask_price'] and market_data['bid_price']:
            print(f"📏 Spread (Écart)             : {round(market_data['ask_price'] - market_data['bid_price'], 4)}$")