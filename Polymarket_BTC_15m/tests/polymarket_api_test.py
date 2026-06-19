import requests
import json
from datetime import datetime, timezone, timedelta
import math

def get_current_15m_slug():
    """Calcule mathématiquement le slug du marché ACTUEL de 15 minutes"""
    now = datetime.now(timezone.utc)
    
    # Arrondi à la tranche de 15 minutes inférieure (Le début du marché)
    # Ex: S'il est 14h26, (26 // 15) * 15 = 15. Donc 14h15.
    minute_start = (now.minute // 15) * 15
    current_15m = now.replace(minute=minute_start, second=0, microsecond=0)
    
    # Conversion en Timestamp UNIX
    timestamp = int(current_15m.timestamp())
    
    slug = f"btc-updown-15m-{timestamp}"
    print(f"⏱️ Début de la tranche actuelle (UTC) : {current_15m.strftime('%H:%M:%S')}")
    print(f"🔗 Slug mathématique généré : {slug}")
    return slug

def snipe_15m_market():
    print("🎯 Démarrage du Sniper Déterministe (15 minutes)...")
    
    slug = get_current_15m_slug()
    
    # 1. Attaque de l'API Gamma avec le Query Parameter (Contournement de l'Erreur 422)
    print("\n🌐 Interrogation de l'API Gamma pour les Token IDs...")
    url_gamma = f"https://gamma-api.polymarket.com/events?slug={slug}"
    
    res_gamma = requests.get(url_gamma)
    if res_gamma.status_code != 200:
        print(f"❌ Erreur Serveur Gamma. (Code {res_gamma.status_code})")
        return
        
    events = res_gamma.json()
    if not events or len(events) == 0:
        print("❌ Marché introuvable sur Gamma. (Il n'est peut-être pas encore ouvert).")
        return
        
    event = events[0] # On prend le premier résultat qui correspond au slug
    print(f"✅ Événement trouvé : {event.get('title')}")
    
    markets = event.get('markets', [])
    if not markets:
        print("❌ Aucun sous-marché disponible.")
        return
        
    # Extraction du Token YES
    clob_tokens = json.loads(markets[0].get('clobTokenIds', '[]'))
    if len(clob_tokens) == 0:
        print("❌ Tokens CLOB introuvables pour ce marché.")
        return
        
    token_yes = clob_tokens[0]
    
    # 2. Interrogation directe du Moteur de Trading (CLOB)
    print("\n⚡ Extraction du Carnet d'Ordres (CLOB)...")
    url_book = f"https://clob.polymarket.com/book?token_id={token_yes}"
    
    res_book = requests.get(url_book)
    if res_book.status_code == 200:
        book = res_book.json()
        bids = book.get('bids', [])
        asks = book.get('asks', [])
        
        if bids and asks:
            best_bid = float(bids[0].get('price'))
            best_ask = float(asks[0].get('price'))
            midpoint = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            
            # Calcul du Volume de liquidité (Top 3)
            bid_vol = sum(float(b.get('size')) for b in bids[:3])
            ask_vol = sum(float(a.get('size')) for a in asks[:3])
            
            print(f"  [PRIX] 🎯 Midpoint : {midpoint:.4f} $")
            print(f"  [LIQUIDITÉ] Acheteurs (Bid) : {best_bid:.4f} $ | Volume dispo: {bid_vol:.2f}")
            print(f"  [LIQUIDITÉ] Vendeurs (Ask)  : {best_ask:.4f} $ | Volume dispo: {ask_vol:.2f}")
            print(f"  [RISQUE] Spread (Friction)  : {spread:.4f} $")
        else:
            print("  ⚠️ Carnet d'ordres complètement vide.")
    else:
        print("❌ Erreur accès CLOB.")

if __name__ == "__main__":
    snipe_15m_market()