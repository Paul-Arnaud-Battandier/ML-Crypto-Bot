"""
select_pair.py
──────────────
Scan hebdomadaire des paires cointégrées.
Sélectionne la meilleure paire et écrit le résultat
dans ../data/best_pair.json.

Peut être appelé :
  - En standalone : python select_pair.py
  - En import     : from select_pair import get_best_pair
"""

import time
import sys
import ccxt
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# ── Chemins ───────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).parent.parent
DATA_DIR      = ROOT_DIR / "data"

# Racine du projet (ML_Crypto_Bot/) — un niveau au-dessus de Regime_Detector/
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from state_store import get_state, set_state
DATA_DIR.mkdir(exist_ok=True)
BEST_PAIR_FILE = DATA_DIR / "best_pair.json"

# ── Configuration ─────────────────────────────────────────────
DAYS      = 90    # Fenêtre de 90j (évite faux positifs bull run)
TIMEFRAME = '1h'  # 1h = meilleur compromis signal/bruit
WINDOW    = 50    # Rolling OLS window

# Paires candidates avec logique économique
CANDIDATE_PAIRS = [
    ('AAVE/USDT', 'ETH/USDT'),   # DeFi vs sous-jacent ← référence
    ('UNI/USDT',  'ETH/USDT'),   # DEX vs ETH
    ('LINK/USDT', 'ETH/USDT'),   # Oracle vs ETH
    ('ARB/USDT',  'ETH/USDT'),   # L2 vs ETH
    ('OP/USDT',   'ETH/USDT'),   # L2 vs ETH
    ('AAVE/USDT', 'UNI/USDT'),   # DeFi intra
    ('SOL/USDT',  'ETH/USDT'),   # L1 vs ETH
    ('AVAX/USDT', 'ETH/USDT'),   # L1 vs ETH
    ('ETH/USDT',  'BTC/USDT'),   # Macro pair
    ('LDO/USDT',  'ETH/USDT'),   # Liquid staking vs ETH
    ('MATIC/USDT','ETH/USDT'),   # L2 vs ETH
]

# Seuils de qualité minimaux
MIN_SCORE       = 30   # Score minimum pour être sélectionnée
MAX_HALF_LIFE_H = 72   # Half-life max en heures (3 jours)
MIN_HALF_LIFE_H = 1    # Half-life min en heures

# ── Data ──────────────────────────────────────────────────────
def fetch_close(symbol, timeframe=TIMEFRAME, days=DAYS):
    exchange = ccxt.binance({'enableRateLimit': True})
    since    = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    all_ohlcv = []
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            tf_ms = {'15m': 15*60*1000, '1h': 3600*1000, '4h': 4*3600*1000}
            if ohlcv[-1][0] >= int(datetime.now().timestamp()*1000) - tf_ms.get(timeframe, 3600*1000):
                break
        except Exception as e:
            print(f'    ⚠️ {symbol}: {e}')
            break
    if not all_ohlcv:
        return None
    df = pd.DataFrame(all_ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')[['close']]
    df = df[~df.index.duplicated(keep='first')]
    df.columns = [symbol]
    return df

# ── Indicateurs ───────────────────────────────────────────────
def compute_hurst(series, max_lag=30):
    try:
        lags = range(2, max_lag)
        tau  = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        if any(t == 0 for t in tau):
            return np.nan
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return float(poly[0])
    except:
        return np.nan

def compute_half_life(spread):
    try:
        lag   = spread.shift(1).dropna()
        delta = spread.diff().dropna()
        lag, delta = lag.align(delta, join='inner')
        reg = OLS(delta, add_constant(lag)).fit()
        if reg.params[0] >= 0:
            return np.inf
        return float(-np.log(2) / reg.params[0])
    except:
        return np.inf

# ── Analyse d'une paire ───────────────────────────────────────
def analyze_pair(sym1, sym2, prices):
    if sym1 not in prices or sym2 not in prices:
        return None

    df = pd.concat([prices[sym1], prices[sym2]], axis=1).dropna()
    if len(df) < WINDOW * 3:
        return None

    split = int(len(df) * 0.8)
    train = df.iloc[:split]

    # Cointégration in-sample
    _, pvalue, _ = coint(train[sym1], train[sym2])

    # Rolling OLS hedge ratio
    rolling_cov = df[sym1].rolling(WINDOW).cov(df[sym2])
    rolling_var = df[sym2].rolling(WINDOW).var()
    ratio       = rolling_cov / rolling_var
    spread      = (df[sym1] - ratio * df[sym2]).dropna()

    # Métriques
    adf_pvalue      = adfuller(spread.iloc[split:])[1]
    hurst           = compute_hurst(spread.values)
    half_life_c     = compute_half_life(spread)  # en bougies

    # Conversion en heures
    tf_hours   = {'15m': 0.25, '1h': 1.0, '4h': 4.0}
    hl_hours   = half_life_c * tf_hours.get(TIMEFRAME, 1.0)

    # Z-Score stats
    z         = (spread - spread.rolling(WINDOW).mean()) / spread.rolling(WINDOW).std()
    z         = z.dropna()
    crossings = int(((z.shift(1) * z) < 0).sum())
    amplitude = float(z[(z > 2) | (z < -2)].abs().mean()) if len(z) > 0 else 0

    # Hedge ratio actuel
    current_ratio = float(ratio.iloc[-1]) if not pd.isna(ratio.iloc[-1]) else None

    return {
        'sym1'         : sym1,
        'sym2'         : sym2,
        'coint_pvalue' : round(pvalue, 5),
        'adf_oos'      : round(adf_pvalue, 5),
        'hurst'        : round(hurst, 4) if not np.isnan(hurst) else None,
        'half_life_h'  : round(hl_hours, 1),
        'zero_cross'   : crossings,
        'amplitude'    : round(amplitude, 3) if not np.isnan(amplitude) else 0,
        'hedge_ratio'  : round(current_ratio, 6) if current_ratio else None,
        'n_candles'    : len(df),
        'timeframe'    : TIMEFRAME,
        'window_days'  : DAYS,
    }

# ── Scoring ───────────────────────────────────────────────────
def score_pair(res):
    score = 0
    if res['hurst'] is None:
        return 0

    # Hurst (35 pts) — critère le plus important
    if   res['hurst'] < 0.30: score += 35
    elif res['hurst'] < 0.35: score += 25
    elif res['hurst'] < 0.40: score += 15
    elif res['hurst'] < 0.50: score += 5

    # Cointégration (20 pts)
    if   res['coint_pvalue'] < 0.01: score += 20
    elif res['coint_pvalue'] < 0.05: score += 12
    elif res['coint_pvalue'] < 0.10: score += 5

    # ADF out-of-sample (20 pts)
    if   res['adf_oos'] < 0.01: score += 20
    elif res['adf_oos'] < 0.05: score += 12
    elif res['adf_oos'] < 0.10: score += 5

    # Half-life (15 pts) — idéal 2-48h
    hl = res['half_life_h']
    if   2  <= hl <= 24: score += 15
    elif 1  <= hl <= 48: score += 8
    elif 0  <  hl <= 72: score += 3

    # Zero-crossings (10 pts) — activité
    if   res['zero_cross'] > 150: score += 10
    elif res['zero_cross'] > 75:  score += 5
    elif res['zero_cross'] > 30:  score += 2

    return score

# ── Fonction principale ───────────────────────────────────────
def get_best_pair(verbose=True):
    """
    Scanne toutes les paires candidates et retourne la meilleure.
    Écrit le résultat dans best_pair.json.

    Returns:
        dict | None
    """
    if verbose:
        print("="*60)
        print("🔍 SCAN HEBDOMADAIRE DES PAIRES")
        print(f"   Timeframe : {TIMEFRAME} | Fenêtre : {DAYS}j")
        print("="*60)

    # 1. Téléchargement
    all_symbols = list(set([s for pair in CANDIDATE_PAIRS for s in pair]))
    prices      = {}

    if verbose:
        print("\n📥 Téléchargement des données...")
    for sym in all_symbols:
        time.sleep(0.5)  # Espace les appels pour éviter de dépasser le rate-limit weight
        df = fetch_close(sym)
        if df is not None and len(df) > WINDOW * 3:
            prices[sym] = df

    # 2. Scan
    if verbose:
        print("\n🔬 Analyse des paires...")
    results = []
    for sym1, sym2 in CANDIDATE_PAIRS:
        res = analyze_pair(sym1, sym2, prices)
        if res is None:
            if verbose:
                print(f"  ⚠️  {sym1} / {sym2} → données insuffisantes")
            continue
        res['score'] = score_pair(res)
        results.append(res)
        if verbose:
            h_ok = '✅' if res['hurst'] and res['hurst'] < 0.5 else '❌'
            c_ok = '✅' if res['coint_pvalue'] < 0.05 else '⚠️'
            print(f"  {c_ok} {sym1:<12}/ {sym2:<12} | "
                  f"score={res['score']:>3} | "
                  f"hurst={h_ok}{res['hurst']} | "
                  f"HL={res['half_life_h']}h | "
                  f"coint={res['coint_pvalue']:.4f}")

    if not results:
        if verbose:
            print("\n❌ Aucune paire valide trouvée.")
        return None

    # 3. Sélection
    results.sort(key=lambda x: x['score'], reverse=True)
    best = results[0]

    # 4. Vérification score minimum
    if best['score'] < MIN_SCORE:
        if verbose:
            print(f"\n⚠️ Meilleur score ({best['score']}) sous le minimum ({MIN_SCORE}).")
            print("   Aucune paire suffisamment fiable — bots désactivés.")
        best['valid'] = False
    elif (best['half_life_h'] > MAX_HALF_LIFE_H or
          best['half_life_h'] < MIN_HALF_LIFE_H):
        if verbose:
            print(f"\n⚠️ Half-life hors plage ({best['half_life_h']}h).")
        best['valid'] = False
    else:
        best['valid'] = True

    best['updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 5. Sauvegarde (Supabase — le fichier local ne survit pas aux redéploiements Render)
    set_state('best_pair', best)

    # 6. Affichage
    if verbose:
        print(f"\n{'='*60}")
        if best['valid']:
            print(f"🏆 MEILLEURE PAIRE : {best['sym1']} / {best['sym2']}")
        else:
            print(f"⚠️  PAIRE PAR DÉFAUT (score faible) : {best['sym1']} / {best['sym2']}")
        print(f"{'='*60}")
        print(f"   Score          : {best['score']}/100")
        print(f"   Hurst          : {best['hurst']}")
        print(f"   Coint p-value  : {best['coint_pvalue']}")
        print(f"   ADF OOS        : {best['adf_oos']}")
        print(f"   Half-Life      : {best['half_life_h']}h")
        print(f"   Zero-Crossings : {best['zero_cross']}")
        print(f"   Hedge Ratio    : {best['hedge_ratio']}")
        print(f"   Valide         : {'✅ Oui' if best['valid'] else '❌ Non'}")
        print(f"\n💾 Sauvegardé dans Supabase (bot_state['best_pair'])")

    return best

# ── Lire depuis Supabase ───────────────────────────────────────
def read_best_pair():
    """
    Lit la meilleure paire depuis Supabase (bot_state['best_pair']).
    À utiliser dans live_bot.py.

    Returns:
        dict | None
    """
    data = get_state('best_pair')
    if data is None:
        print("⚠️ Aucun scan disponible — lance select_pair.py d'abord")
        return None
    try:
        ts  = datetime.strptime(data['updated'], '%Y-%m-%d %H:%M:%S')
        age = (datetime.now() - ts).total_seconds() / 3600 / 24
        if age > 8:
            print(f"⚠️ Scan obsolète ({age:.1f}j) — relance select_pair.py")
    except Exception:
        pass
    return data

if __name__ == "__main__":
    get_best_pair(verbose=True)