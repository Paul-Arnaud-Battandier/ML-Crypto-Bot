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

── Source de données : Kraken (pas Binance) ────────────────────
Binance bloque (HTTP 451) toutes les IP de datacenters US, ce qui
inclut les runners GitHub Actions. Ce script ne fait que lire des
prix publics (pas de compte, pas d'ordre) — n'importe quel exchange
liquide donne des indicateurs équivalents. Kraken n'a pas ce
blocage géographique, donc on l'utilise ici.
⚠️ Les bots de trading (StatArb, Funding Carry) restent sur Binance
Demo Trading — ce changement ne concerne QUE la lecture de prix
pour le calcul du régime.

── Script one-shot (GitHub Actions) ────────────────────────────
Ce script est lancé sur une machine neuve à chaque exécution (cron
GitHub Actions, toutes les heures). Il ne garde aucun état local —
tout est lu/écrit dans Supabase via state_store.py :
  - bot_state['current_regime']  → dernier résultat (lu par les autres bots)
  - regime_history (table)       → historique complet (pour le dashboard)

Appel : python compute_regime.py
"""

import sys
import ccxt
import pandas as pd
import numpy as np
import requests as _requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # ML_Crypto_Bot/
from state_store import set_state
import os

# ── Seuils validés ────────────────────────────────────────────
BTC_VOL_THRESHOLD   = 0.60
ETH_HURST_MEAN_REV  = 0.43
ETH_HURST_TREND     = 0.47
ETH_ADX_MEAN_REV    = 36
ETH_ADX_TREND       = 28

HURST_WINDOW = 100
ADX_PERIOD   = 14
VOL_WINDOW   = 24

STRATEGY_MAP = {
    'MEAN_REV' : {'stat_arb': True,  'funding_carry': False, 'momentum': False},
    'TRENDING' : {'stat_arb': False, 'funding_carry': False, 'momentum': True},
    'HIGH_VOL' : {'stat_arb': False, 'funding_carry': True,  'momentum': False},
    'NEUTRAL'  : {'stat_arb': False, 'funding_carry': True,  'momentum': False},
}

REGIME_EMOJI = {'MEAN_REV': '🟢', 'TRENDING': '🟡', 'HIGH_VOL': '🔴', 'NEUTRAL': '⚪'}

_SB_URL = os.getenv('SUPABASE_URL', '')
_SB_KEY = os.getenv('SUPABASE_KEY', '')


def fetch_ohlcv(exchange, symbol, limit=150):
    ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df.set_index('timestamp')


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
    up, down = high - high.shift(1), low.shift(1) - low
    plus_dm  = np.where((up > down) & (up > 0),   up,   0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_di  = 100 * pd.Series(plus_dm,  index=df.index).ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean() / atr
    dx  = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).fillna(0)
    return float(dx.ewm(span=period, adjust=False).mean().iloc[-1])


def compute_realized_vol(close, window=24):
    returns = close.pct_change()
    vol = returns.rolling(window).std() * np.sqrt(window * 365)
    return float(vol.iloc[-1])


def classify_regime(btc_vol, eth_hurst, eth_adx):
    if btc_vol > BTC_VOL_THRESHOLD:
        return 'HIGH_VOL'
    if not np.isnan(eth_hurst):
        if eth_hurst > ETH_HURST_TREND and eth_adx > ETH_ADX_TREND:
            return 'TRENDING'
        elif eth_hurst < ETH_HURST_MEAN_REV and eth_adx < ETH_ADX_MEAN_REV:
            return 'MEAN_REV'
    return 'NEUTRAL'


def _log_regime_history(result):
    """Historique complet (table séparée, pour le dashboard)."""
    if not _SB_URL or not _SB_KEY:
        return
    try:
        _requests.post(
            f"{_SB_URL}/rest/v1/regime_history",
            headers={
                'apikey': _SB_KEY, 'Authorization': f'Bearer {_SB_KEY}',
                'Content-Type': 'application/json', 'Prefer': 'return=minimal',
            },
            json={
                'timestamp': result['timestamp'], 'regime': result['regime'],
                'btc_vol': result['btc_vol'], 'eth_hurst': result['eth_hurst'],
                'eth_adx': result['eth_adx'], 'btc_price': result['btc_price'],
                'eth_price': result['eth_price'],
            },
            timeout=10,
        )
    except Exception as e:
        print(f"⚠️ Erreur log regime_history : {e}")


def get_current_regime(verbose=True):
    """
    Calcule le régime actuel (BTC vol + ETH hurst/ADX), l'écrit dans
    bot_state['current_regime'] (Supabase) et dans regime_history (log).
    """
    exchange = ccxt.kraken({'enableRateLimit': True})

    df_btc = fetch_ohlcv(exchange, 'BTC/USDT', limit=150)
    df_eth = fetch_ohlcv(exchange, 'ETH/USDT', limit=150)

    btc_vol   = compute_realized_vol(df_btc['close'], window=VOL_WINDOW)
    btc_price = float(df_btc['close'].iloc[-1])
    eth_hurst = compute_hurst(df_eth['close'].values[-HURST_WINDOW:])
    eth_adx   = compute_adx(df_eth, period=ADX_PERIOD)
    eth_price = float(df_eth['close'].iloc[-1])

    regime     = classify_regime(btc_vol, eth_hurst, eth_adx)
    strategies = STRATEGY_MAP[regime]

    result = {
        'regime'     : regime,
        'btc_vol'    : round(btc_vol, 4),
        'btc_price'  : round(btc_price, 2),
        'eth_hurst'  : round(eth_hurst, 4) if not np.isnan(eth_hurst) else None,
        'eth_adx'    : round(eth_adx, 2),
        'eth_price'  : round(eth_price, 2),
        'strategies' : strategies,
        'timestamp'  : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    set_state('current_regime', result)
    _log_regime_history(result)

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
        print(f"  ⏰ {result['timestamp']}")

    return result


if __name__ == "__main__":
    get_current_regime(verbose=True)