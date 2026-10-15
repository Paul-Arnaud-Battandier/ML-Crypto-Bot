"""
check_kraken_pairs.py — LECTURE SEULE, aucune clé API requise.
───────────────────────────────────────────────────────────────
Avant de migrer select_pair.py de Binance vers Kraken, on vérifie deux
choses qu'on ne peut pas deviner :

  1. COUVERTURE : Kraken liste-t-il chacun des tokens candidats ?
     (ARB, OP, LDO sont moins courants ; MATIC a été renommé POL)
  2. PROFONDEUR : combien de bougies 1h Kraken renvoie-t-il réellement
     quand on demande 90 jours ? (select_pair.py en utilise 90j ; Kraken
     plafonnerait à ~720 bougies = ~30j, à confirmer par ce test)

Usage : python check_kraken_pairs.py
"""

import time
import ccxt
from datetime import datetime, timedelta

BASES = ['AAVE', 'ETH', 'UNI', 'LINK', 'ARB', 'OP', 'SOL', 'AVAX', 'BTC', 'LDO', 'MATIC', 'POL']
QUOTES = ['USDT', 'USD']  # on teste les deux : USD est souvent plus liquide sur Kraken
DEPTH_TEST_DAYS = 90


def find_symbol(exchange, base):
    for quote in QUOTES:
        sym = f"{base}/{quote}"
        if sym in exchange.markets:
            return sym
    return None


def main():
    exchange = ccxt.kraken({'enableRateLimit': True})
    print("🔍 Chargement des marchés Kraken (lecture seule)...")
    exchange.load_markets()

    print(f"\n{'='*64}")
    print("  1. COUVERTURE")
    print(f"{'='*64}")
    found = {}
    for base in BASES:
        sym = find_symbol(exchange, base)
        found[base] = sym
        print(f"  {base:<6} → {sym if sym else '❌ absent de Kraken'}")

    print(f"\n{'='*64}")
    print(f"  2. PROFONDEUR D'HISTORIQUE 1h (demande : {DEPTH_TEST_DAYS} jours)")
    print(f"{'='*64}")
    since_ms = int((datetime.now() - timedelta(days=DEPTH_TEST_DAYS)).timestamp() * 1000)
    for base in ('ETH', 'AAVE'):
        sym = found.get(base)
        if not sym:
            print(f"  {base} : non testable (absent)")
            continue
        try:
            ohlcv = exchange.fetch_ohlcv(sym, '1h', since=since_ms, limit=1000)
        except Exception as e:
            print(f"  {sym} : erreur {e}")
            continue
        if not ohlcv:
            print(f"  {sym} : 0 bougie renvoyée")
            continue
        first = datetime.fromtimestamp(ohlcv[0][0] / 1000)
        last = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
        span_days = (last - first).total_seconds() / 86400
        print(f"  {sym} : {len(ohlcv)} bougies | {first:%Y-%m-%d} → {last:%Y-%m-%d} "
              f"(~{span_days:.0f} jours effectifs)")
        time.sleep(1.5)

    print(f"\n{'='*64}")
    print("  RÉSUMÉ")
    print(f"{'='*64}")
    missing = [b for b, s in found.items() if s is None and b not in ('MATIC', 'POL')]
    print(f"  Tokens absents (hors MATIC/POL) : {missing if missing else 'aucun ✅'}")
    print(f"  MATIC : {found.get('MATIC') or 'absent'} | POL : {found.get('POL') or 'absent'}")
    print("\n  Envoie-moi cette sortie complète : elle décide si la migration est")
    print("  viable telle quelle ou s'il faut réduire la liste de paires / la fenêtre.")


if __name__ == "__main__":
    main()