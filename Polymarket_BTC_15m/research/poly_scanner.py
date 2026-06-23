import requests

def analyser_carnet_precis():
    # Le Token ID exact que ton diagnostic a trouvé
    token_id = "114793892740393962626630492624877542882946300759236562661703170722171757805063"
    url = f"https://clob.polymarket.com/book?token_id={token_id}"
    
    print("="*60)
    print("🔎 ANALYSE DU CARNET : BITCOIN 4 PM ET")
    print("="*60)
    
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        book = resp.json()
        
        bids = book.get('bids', [])
        asks = book.get('asks', [])
        
        if not bids or not asks:
            print("⚠️ Le carnet manque de Bids ou d'Asks.")
            return

        # Top of Book (Meilleurs prix)
        best_bid = float(bids[0]['price'])
        best_bid_size = float(bids[0]['size'])
        best_ask = float(asks[0]['price'])
        best_ask_size = float(asks[0]['size'])
        
        spread = best_ask - best_bid
        
        # Volumes totaux
        total_bid_vol = sum(float(b['size']) for b in bids)
        total_ask_vol = sum(float(a['size']) for a in asks)
        
        print(f"Meilleur Acheteur (BID) : {best_bid:.3f} $  (Veut acheter {best_bid_size:.0f} parts)")
        print(f"Meilleur Vendeur  (ASK) : {best_ask:.3f} $  (Veut vendre {best_ask_size:.0f} parts)")
        print("-" * 60)
        print(f"🔥 SPREAD ACTUEL       : {spread:.3f} $")
        print("-" * 60)
        print(f"Volume total caché derrière : {total_bid_vol:.0f} parts à l'achat / {total_ask_vol:.0f} parts à la vente")
        
        if spread <= 0.05:
            print("\n✅ VERDICT : LIQUIDITÉ EXCELLENTE. FEU VERT POUR LE BOT !")
        elif spread <= 0.10:
            print("\n🟠 VERDICT : LIQUIDITÉ MOYENNE. Tradeable mais avec prudence.")
        else:
            print("\n❌ VERDICT : SPREAD TROP LARGE. Risque de perte mathématique élevé.")
            
    except Exception as e:
        print(f"Erreur lors de la lecture du carnet : {e}")

if __name__ == "__main__":
    analyser_carnet_precis()