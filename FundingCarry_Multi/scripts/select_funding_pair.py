"""
select_funding_pair.py
───────────────────────
Scan hebdomadaire des meilleures opportunités de Funding Rate Carry.
Filtre strict basé sur :
  - APR moyen 30j > seuil (rentabilité après frais)
  - % de paiements positifs > seuil (consistance)
  - Funding actuel positif (ne pas entrer dans le mauvais sens)

Écrit le résultat dans Supabase (bot_state['best_funding']).
"""

import ccxt
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import re
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# ── Détection de ban Binance robuste (même fix que select_pair.py, 14/09) ──
# ccxt classe l'erreur -1003 en RateLimitExceeded, pas DDoSProtection —
# notre ancien `except ccxt.DDoSProtection` ne l'attrapait jamais.
_RATE_LIMIT_EXC_TYPES = tuple(
    cls for cls in (
        getattr(ccxt, 'DDoSProtection', None),
        getattr(ccxt, 'RateLimitExceeded', None),
    ) if cls is not None
)

def _is_rate_limit_error(exc):
    if isinstance(exc, _RATE_LIMIT_EXC_TYPES):
        return True
    msg = str(exc)
    return '-1003' in msg or 'Way too many requests' in msg or 'banned until' in msg

def _extract_ban_until_ms(exc):
    m = re.search(r'banned until (\d+)', str(exc))
    return int(m.group(1)) if m else None

def _wait_out_ban(exc, max_wait_seconds=420):
    """Attend l'expiration du ban plutôt que d'abandonner le scan (plafonné
    pour rester sous le timeout du subprocess, 600s dans background_jobs.py)."""
    ban_until_ms = _extract_ban_until_ms(exc)
    if ban_until_ms is None:
        return False
    wait_s = (ban_until_ms / 1000) - time.time() + 2
    if wait_s <= 0:
        return True
    if wait_s > max_wait_seconds:
        print(f"    ⏳ Ban trop long ({wait_s:.0f}s > {max_wait_seconds}s max) — abandon pour cette exécution.")
        return False
    print(f"    ⏳ Ban Binance détecté — attente de {wait_s:.0f}s avant de reprendre...")
    time.sleep(wait_s)
    return True

# ── Chemins ───────────────────────────────────────────────────
ROOT_DIR          = Path(__file__).parent.parent
DATA_DIR          = ROOT_DIR / "data"

# Racine du projet (ML_Crypto_Bot/) — un niveau au-dessus de FundingCarry_Multi/
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from state_store import get_state, set_state
DATA_DIR.mkdir(exist_ok=True)
BEST_FUNDING_FILE = DATA_DIR / "best_funding.json"

# ── Paramètres ────────────────────────────────────────────────
BNB_DISCOUNT  = False   # True si tu as activé le paiement en BNB (-25%)
SPOT_FEE      = 0.0010  # 0.10% standard
PERP_FEE      = 0.0005  # 0.05% standard

# Frais round-trip (entrée + sortie, les 2 pattes)
ROUND_TRIP_FEE = (SPOT_FEE + PERP_FEE) * 2 * (0.75 if BNB_DISCOUNT else 1.0)

# Filtres de sélection
MIN_APR_30D       = 4.0    # % APR minimum sur 30j (breakeven ~25j standard)
MIN_POSITIVE_PCT  = 60.0   # % minimum de paiements positifs sur 30j
MIN_CURRENT_APR   = 0.0    # Funding actuel doit être positif

# Délai entre appels API — augmenté de 0.5s à 1.2s suite à un ban IP
# Binance (erreur 418 / -1003 "Way too many requests") pendant un rescan
# du 04/09 : 20 symboles x 2 appels (history + current) à 0.5s d'écart
# dépassait le rate-limit de poids malgré enableRateLimit=True (qui
# espace les requêtes mais ne garantit pas de rester sous le seuil).
API_CALL_DELAY = 1.2

UNIVERSE = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
    'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT',
    'MATIC/USDT', 'LTC/USDT', 'UNI/USDT', 'AAVE/USDT', 'ARB/USDT',
    'OP/USDT', 'SUI/USDT', 'NEAR/USDT', 'APT/USDT', 'FIL/USDT',
]


def fetch_funding_data(exchange, symbol, days=30):
    """Récupère le funding actuel + historique 30j"""
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    try:
        # Historique
        history = exchange.fetch_funding_rate_history(symbol, since=since, limit=300)
        if not history:
            return None, None
        rates = [h['fundingRate'] for h in history]

        time.sleep(API_CALL_DELAY)  # espace les 2 appels du même symbole

        # Actuel
        current = exchange.fetch_funding_rate(symbol)
        current_rate = current.get('fundingRate', None)

        return rates, current_rate
    except Exception as e:
        if _is_rate_limit_error(e):
            print(f"    🚫 {symbol}: rate-limit/ban Binance détecté ({e})")
            raise
        print(f"    ⚠️  {symbol}: {e}")
        return None, None


def compute_breakeven_days(apr_30d):
    """Nombre de jours de holding minimum pour couvrir les frais"""
    if apr_30d <= 0:
        return float('inf')
    daily_pct = apr_30d / 100 / 365
    return ROUND_TRIP_FEE / daily_pct


def score_opportunity(row):
    """Score composite 0-100"""
    score = 0
    # APR moyen 30j (40 pts) — critère principal
    if   row['apr_30d'] >= 8.0: score += 40
    elif row['apr_30d'] >= 6.0: score += 30
    elif row['apr_30d'] >= 4.0: score += 20

    # Consistance (30 pts)
    if   row['positive_pct'] >= 80: score += 30
    elif row['positive_pct'] >= 70: score += 20
    elif row['positive_pct'] >= 60: score += 10

    # Funding actuel (20 pts) — aligné avec le signal entrée
    if   row['current_apr'] >= 6.0: score += 20
    elif row['current_apr'] >= 3.0: score += 12
    elif row['current_apr'] >= 0.0: score += 5

    # Breakeven court (10 pts)
    be = row['breakeven_days']
    if   be <= 15: score += 10
    elif be <= 25: score += 5

    return score


def get_best_funding(verbose=True):
    """
    Scanne toutes les paires candidates et retourne la meilleure
    opportunité de carry (Long Spot + Short Perp).
    Écrit le résultat dans Supabase (bot_state['best_funding']).
    """
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })

    if verbose:
        print("=" * 60)
        print("💰 SCAN FUNDING RATE CARRY")
        print(f"   Seuil APR 30j   : >{MIN_APR_30D}%")
        print(f"   Frais round-trip : {ROUND_TRIP_FEE*100:.3f}% {'(BNB)' if BNB_DISCOUNT else '(standard)'}")
        print("=" * 60)

    results = []
    remaining = list(UNIVERSE)
    while remaining:
        sym = remaining[0]
        time.sleep(API_CALL_DELAY)  # Espace les appels pour éviter de dépasser le rate-limit weight
        try:
            rates, current_rate = fetch_funding_data(exchange, sym)
        except Exception as e:
            if _is_rate_limit_error(e):
                if _wait_out_ban(e):
                    continue  # on retente CE symbole après l'attente
                print("\n🚫 Scan interrompu à cause d'un ban/rate-limit Binance trop long — "
                      "résultat basé uniquement sur les symboles déjà scannés avant le ban.")
                break
            raise
        if rates is None or current_rate is None:
            remaining.pop(0)
            continue

        avg_rate     = np.mean(rates)
        positive_pct = np.mean([r > 0 for r in rates]) * 100
        apr_30d      = avg_rate * 3 * 365 * 100       # annualisé en %
        current_apr  = current_rate * 3 * 365 * 100
        breakeven    = compute_breakeven_days(apr_30d)

        result = {
            'symbol'       : sym,
            'current_rate' : round(current_rate * 100, 4),  # % par paiement
            'current_apr'  : round(current_apr, 2),
            'apr_30d'      : round(apr_30d, 2),
            'positive_pct' : round(positive_pct, 1),
            'breakeven_days': round(breakeven, 1),
            'n_payments'   : len(rates),
        }
        result['score'] = score_opportunity(result)
        results.append(result)

        if verbose:
            valid = '✅' if (apr_30d >= MIN_APR_30D and
                             positive_pct >= MIN_POSITIVE_PCT and
                             current_apr >= MIN_CURRENT_APR) else '⚠️ '
            print(f"  {valid} {sym:<12} | APR 30j: {apr_30d:>7.2f}% | "
                  f"Actuel: {current_apr:>7.2f}% | "
                  f"Positif: {positive_pct:.0f}% | "
                  f"Breakeven: {breakeven:.0f}j")
        remaining.pop(0)

    if not results:
        print("❌ Aucune donnée récupérée")
        return None

    df = pd.DataFrame(results).sort_values('score', ascending=False)

    # Filtrage strict
    df_valid = df[
        (df['apr_30d']      >= MIN_APR_30D) &
        (df['positive_pct'] >= MIN_POSITIVE_PCT) &
        (df['current_apr']  >= MIN_CURRENT_APR)
    ]

    if verbose:
        print()

    if df_valid.empty:
        best = df.iloc[0].to_dict()
        best['valid'] = False
        if verbose:
            print(f"⚠️  Aucune paire ne passe les filtres strictes.")
            print(f"   Meilleure disponible : {best['symbol']} (APR 30j: {best['apr_30d']}%)")
            print(f"   Bots désactivés — pas assez rentable après frais.")
    else:
        best = df_valid.iloc[0].to_dict()
        best['valid'] = True
        if verbose:
            print(f"🏆 MEILLEURE OPPORTUNITÉ : {best['symbol']}")
            print(f"   APR 30j      : {best['apr_30d']}%")
            print(f"   APR actuel   : {best['current_apr']}%")
            print(f"   Positif 30j  : {best['positive_pct']}%")
            print(f"   Breakeven    : {best['breakeven_days']} jours")
            print(f"   Score        : {best['score']}/100")

    best['updated']     = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    best['bnb_discount']= BNB_DISCOUNT
    best['round_trip_fee_pct'] = round(ROUND_TRIP_FEE * 100, 3)

    # Sauvegarde (Supabase — le fichier local ne survit pas aux redéploiements Render)
    set_state('best_funding', best)

    if verbose:
        print(f"\n💾 Sauvegardé dans Supabase (bot_state['best_funding'])")

    return best


def read_best_funding():
    """Lit la meilleure opportunité depuis Supabase (bot_state['best_funding'])"""
    data = get_state('best_funding')
    if data is None:
        print("⚠️  Aucun scan funding — lance select_funding_pair.py d'abord")
        return None
    try:
        ts  = datetime.strptime(data['updated'], '%Y-%m-%d %H:%M:%S')
        age = (datetime.now() - ts).total_seconds() / 3600 / 24
        if age > 8:
            print(f"⚠️  Scan funding obsolète ({age:.1f}j) — relance select_funding_pair.py")
    except Exception:
        pass
    return data


if __name__ == "__main__":
    get_best_funding(verbose=True)