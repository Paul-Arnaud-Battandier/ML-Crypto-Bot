import requests

def trouver_marches_directionnels():
    print("="*80)
    print("📈 RECHERCHE DES MARCHÉS DIRECTIONNELS (UP/DOWN) LES PLUS LIQUIDES")
    print("="*80)
    
    url = "https://gamma-api.polymarket.com/events?active=true&closed=false&limit=1000"
    
    print("1️⃣ Téléchargement des marchés en cours...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        events = response.json()
    except Exception as e:
        print(f"Erreur Gamma API: {e}")
        return

    marches_algo = []
    
    # On veut un sous-jacent crypto ET une structure directionnelle
    assets = ['bitcoin', 'btc', 'eth', 'ethereum', 'solana']
    structures = ['up or down', 'daily close', 'weekly close', 'price at']
    
    for event in events:
        title = event.get('title', '').lower()
        
        # Le titre doit contenir un asset ET une structure de prix
        has_asset = any(a in title for a in assets)
        has_structure = any(s in title for s in structures)
        
        # On exclut les paris événementiels long terme (hit, ETF, etc.)
        is_event_bet = any(bad in title for bad in ['hit', 'etf', 'election', 'approve'])
        
        if has_asset and has_structure and not is_event_bet:
            try:
                liquidity = float(event.get('liquidity', 0) or 0)
                volume = float(event.get('volume', 0) or 0)
            except:
                liquidity, volume = 0, 0
                
            if liquidity > 0:  # On prend tout ce qui a un minimum d'argent
                marches_algo.append({
                    'title': event.get('title'),
                    'slug': event.get('slug'),
                    'liquidity': liquidity,
                    'volume': volume
                })

    marches_algo.sort(key=lambda x: x['liquidity'], reverse=True)
    
    print(f"2️⃣ Analyse terminée. Top des marchés Algo/Directionnels :\n")
    print(f"{'LIQUIDITÉ ($)':<13} | {'VOLUME ($)':<12} | {'TITRE DU MARCHÉ'}")
    print("-" * 80)
    
    for m in marches_algo[:15]:
        print(f"{m['liquidity']:<13,.0f} | {m['volume']:<12,.0f} | {m['title']}")
        print(f"  -> Slug: {m['slug']}\n")

if __name__ == "__main__":
    trouver_marches_directionnels()