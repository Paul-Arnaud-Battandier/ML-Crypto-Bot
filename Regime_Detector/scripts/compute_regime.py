"""
compute_regime.py
─────────────────
Détecteur de régime à deux niveaux :
  - BTC/USDT 1h → détecte HIGH_VOL (stress macro systémique)
  - ETH/USDT 1h → détecte TRENDING / MEAN_REV (dynamique des altcoins DeFi/L2)

Logique :
  1. Si BTC vol > seuil         → HIGH_VOL  (stress global, funding carry)
  2. Si ETH hurst > 0.47 + ADX  → TRENDING  (momentum)
  3. Si ETH hurst < 0.43 + ADX  → MEAN_REV  (stat arb)
  4. Sinon                       → NEUTRAL

Peut être appelé :
  - En standalone : python compute_regime.py
  - En import     : from compute_regime import get_current_regime, read_regime
"""

import ccxt
import requests as _requests
from dotenv import load_dotenv
load_dotenv()
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path

# ── Chemins ───────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent.parent
DATA_DIR    = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
REGIME_FILE = DATA_DIR / "current_regime.json"

# ── Seuils validés ────────────────────────────────────────────
# BTC → stress macro
BTC_VOL_THRESHOLD   = 0.60   # Vol annualisée BTC au-dessus = HIGH_VOL

# ETH → dynamique altcoins
ETH_HURST_MEAN_REV  = 0.43   # En-dessous = mean-reversion
ETH_HURST_TREND     = 0.47   # Au-dessus  = tendance
ETH_ADX_MEAN_REV    = 36     # En-dessous = pas de tendance forte
ETH_ADX_TREND       = 28     # Au-dessus  = signal de tendance

# Paramètres de calcul
HURST_WINDOW = 100   # 100 bougies 1h = ~4 jours
ADX_PERIOD   = 14
VOL_WINDOW   = 24    # 24h pour la vol réalisée

# ── Stratégies actives par régime ─────────────────────────────
STRATEGY_MAP = {
    'MEAN_REV' : {'stat_arb': True,  'funding_carry': False, 'momentum': False},
    'TRENDING' : {'stat_arb': False, 'funding_carry': False, 'momentum': True},
    'HIGH_VOL' : {'stat_arb': False, 'funding_carry': True,  'momentum': False},
    'NEUTRAL'  : {'stat_arb': False, 'funding_carry': True,  'momentum': False},
}

REGIME_EMOJI = {
    'MEAN_REV': '🟢', 'TRENDING': '🟡',
    'HIGH_VOL': '🔴', 'NEUTRAL' : '⚪',
}

# ── Fetch ─────────────────────────────────────────────────────
# ── Fetch ─────────────────────────────────────────────────────
# Instance unique réutilisée — en créer une nouvelle à chaque appel
# accumule des connexions HTTP non fermées (fuite mémoire lente).
_exchange = ccxt.binance({'enableRateLimit': True})

def fetch_ohlcv(symbol, limit=150):
    ohlcv = _exchange.fetch_ohlcv(symbol, '1h', limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    return df

# ── Indicateurs ───────────────────────────────────────────────
def compute_hurst(series, max_lag=20):
    try:
        lags = range(2, max_lag)
        tau  = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        if any(t == 0 for t in tau):
            return np.nan
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return float(poly[0])
    except:
        return np.nan

def compute_adx(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()

    up   = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm  = np.where((up > down) & (up > 0),   up,   0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    plus_di  = 100 * pd.Series(plus_dm,  index=df.index).ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean() / atr
    dx  = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).fillna(0)
    adx = dx.ewm(span=period, adjust=False).mean()
    return float(adx.iloc[-1])

def compute_realized_vol(close, window=24):
    returns = close.pct_change()
    vol = returns.rolling(window).std() * np.sqrt(window * 365)
    return float(vol.iloc[-1])

# ── Classifieur à deux niveaux ────────────────────────────────
def classify_regime(btc_vol, eth_hurst, eth_adx):
    """
    Niveau 1 — BTC : stress macro ?
    Niveau 2 — ETH : trending ou mean-reverting ?
    """
    # Niveau 1 : BTC vol → HIGH_VOL systémique
    if btc_vol > BTC_VOL_THRESHOLD:
        return 'HIGH_VOL'

    # Niveau 2 : ETH dicte la dynamique des altcoins
    if not np.isnan(eth_hurst):
        if eth_hurst > ETH_HURST_TREND and eth_adx > ETH_ADX_TREND:
            return 'TRENDING'
        elif eth_hurst < ETH_HURST_MEAN_REV and eth_adx < ETH_ADX_MEAN_REV:
            return 'MEAN_REV'

    return 'NEUTRAL'

# ── Fonction principale ───────────────────────────────────────
def get_current_regime(verbose=True):
    """
    Calcule et retourne le régime actuel (BTC vol + ETH hurst/ADX).
    Écrit le résultat dans current_regime.json.

    Returns:
        dict : {regime, btc_vol, eth_hurst, eth_adx, strategies, ...}
    """
    # 1. Données BTC et ETH
    df_btc = fetch_ohlcv('BTC/USDT', limit=150)
    df_eth = fetch_ohlcv('ETH/USDT', limit=150)

    # 2. Indicateurs BTC (stress macro)
    btc_vol   = compute_realized_vol(df_btc['close'], window=VOL_WINDOW)
    btc_price = float(df_btc['close'].iloc[-1])

    # 3. Indicateurs ETH (dynamique altcoins)
    eth_hurst = compute_hurst(df_eth['close'].values[-HURST_WINDOW:])
    eth_adx   = compute_adx(df_eth, period=ADX_PERIOD)
    eth_price = float(df_eth['close'].iloc[-1])

    # 4. Classification
    regime     = classify_regime(btc_vol, eth_hurst, eth_adx)
    strategies = STRATEGY_MAP[regime]

    # 5. Résultat
    result = {
        'regime'      : regime,
        # BTC
        'btc_vol'     : round(btc_vol, 4),
        'btc_price'   : round(btc_price, 2),
        # ETH
        'eth_hurst'   : round(eth_hurst, 4) if not np.isnan(eth_hurst) else None,
        'eth_adx'     : round(eth_adx, 2),
        'eth_price'   : round(eth_price, 2),
        # Stratégies
        'strategies'  : strategies,
        'timestamp'   : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    # 6. Sauvegarde
    with open(REGIME_FILE, 'w') as f:
        json.dump(result, f, indent=2)
    _supabase_log_regime(result)

    # 7. Affichage
    if verbose:
        print(f"\n{'='*55}")
        print(f"🎯 RÉGIME ACTUEL : {REGIME_EMOJI[regime]} {regime}")
        print(f"{'='*55}")
        print(f"  📊 BTC  — Vol réalisée : {btc_vol:.4f} "
              f"{'🔴 > seuil' if btc_vol > BTC_VOL_THRESHOLD else '✅ normal'}")
        print(f"  📊 ETH  — Hurst        : {result['eth_hurst']} "
              f"{'🟢 MR' if eth_hurst < ETH_HURST_MEAN_REV else '🟡 TR' if eth_hurst > ETH_HURST_TREND else '⚪'}")
        print(f"  📊 ETH  — ADX          : {eth_adx:.2f}")
        print(f"  💰 BTC ${btc_price:,.2f} | ETH ${eth_price:,.2f}")
        print(f"\n  📋 STRATÉGIES :")
        for strat, active in strategies.items():
            print(f"     {'✅' if active else '❌'} {strat}")
        print(f"\n  💾 {REGIME_FILE}")
        print(f"  ⏰ {result['timestamp']}")

    return result

# ── Lire depuis JSON ──────────────────────────────────────────
def read_regime():
    """
    Lit le dernier régime calculé depuis le fichier JSON.
    À utiliser dans live_bot.py (évite de recalculer à chaque bougie).

    Returns:
        dict | None
    """
    try:
        with open(REGIME_FILE) as f:
            data = json.load(f)
        ts  = datetime.strptime(data['timestamp'], '%Y-%m-%d %H:%M:%S')
        age = (datetime.now() - ts).total_seconds() / 3600
        if age > 2:
            print(f"⚠️  Régime obsolète ({age:.1f}h) — recalcul recommandé")
        return data
    except FileNotFoundError:
        print("⚠️  Aucun régime calculé — lance compute_regime.py d'abord")
        return None

# ── Supabase logging ──────────────────────────────────────────
_SB_URL = os.getenv('SUPABASE_URL', '')
_SB_KEY = os.getenv('SUPABASE_KEY', '')

def _supabase_log_regime(result):
    if not _SB_URL or not _SB_KEY:
        return
    try:
        _requests.post(
            f"{_SB_URL}/rest/v1/regime_history",
            headers={
                'apikey': _SB_KEY,
                'Authorization': f'Bearer {_SB_KEY}',
                'Content-Type': 'application/json',
                'Prefer': 'return=minimal',
            },
            json={
                'timestamp': result['timestamp'],
                'regime'   : result['regime'],
                'btc_vol'  : result['btc_vol'],
                'eth_hurst': result['eth_hurst'],
                'eth_adx'  : result['eth_adx'],
                'btc_price': result['btc_price'],
                'eth_price': result['eth_price'],
            },
            timeout=3,
        )
    except:
        pass

if __name__ == "__main__":
    get_current_regime(verbose=True)