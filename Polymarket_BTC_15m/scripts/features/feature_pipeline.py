"""
feature_pipeline.py
-------------------
Feature engineering complet pour le modèle BTC up/down 15m.
Inspiré de "Advances in Financial Machine Learning" (Lopez de Prado).

Input  : DataFrame OHLCV 1m (colonnes: timestamp, open, high, low, close, volume)
Output : DataFrame de features, une ligne = un point de décision (toutes les 15m)

Organisation des features :
  A. Prix & returns           — base, fractional diff
  B. Volatilité               — réalisée, ratio court/long, ATR
  C. Momentum                 — ROC multi-échelle, autocorrélation
  D. Volume & order flow      — imbalance, VWAP deviation
  E. Microstructure           — Garman-Klass, spread estimé
  F. Features 5m et 15m       — contexte multi-timeframe
  G. Entropy                  — Shannon entropy des returns (LdP)
"""

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# =============================================================================
# A. PRIX & RETURNS
# =============================================================================

def compute_returns(close: pd.Series) -> pd.DataFrame:
    """Returns log et simples sur plusieurs horizons."""
    df = pd.DataFrame(index=close.index)
    df["ret_1m"]  = np.log(close / close.shift(1))
    df["ret_3m"]  = np.log(close / close.shift(3))
    df["ret_5m"]  = np.log(close / close.shift(5))
    df["ret_10m"] = np.log(close / close.shift(10))
    df["ret_15m"] = np.log(close / close.shift(15))
    return df


def fractional_diff(series: pd.Series, d: float = 0.4, thresh: float = 1e-4) -> pd.Series:
    """
    Différentiation fractionnaire (Lopez de Prado, Ch.5).

    Rend la série stationnaire tout en préservant la mémoire long-terme.
    d=0 → pas de diff, d=1 → diff classique, d=0.4 → compromis optimal.

    L'idée : au lieu de différencier entièrement (ce qui détruit la mémoire),
    on applique un opérateur fractionnaire qui décroît exponentiellement.
    """
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
        k += 1

    w = np.array(w[::-1])
    n = len(series)
    result = np.full(n, np.nan)

    for i in range(len(w) - 1, n):
        window = series.iloc[i - len(w) + 1: i + 1].values
        if not np.isnan(window).any():
            result[i] = np.dot(w, window)

    return pd.Series(result, index=series.index, name=f"frac_diff_d{d}")


# =============================================================================
# B. VOLATILITÉ
# =============================================================================

def compute_volatility(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Volatilité réalisée sur plusieurs fenêtres + ratio + ATR.
    """
    close = df_1m["close"]
    high  = df_1m["high"]
    low   = df_1m["low"]
    ret   = np.log(close / close.shift(1))

    out = pd.DataFrame(index=df_1m.index)

    # Volatilité réalisée (écart-type des returns)
    out["vol_5m"]  = ret.rolling(5).std()
    out["vol_15m"] = ret.rolling(15).std()
    out["vol_30m"] = ret.rolling(30).std()

    # Ratio court/long : > 1 = volatilité en hausse (breakout potentiel)
    out["vol_ratio"] = out["vol_5m"] / (out["vol_15m"] + 1e-10)

    # ATR 14 (Average True Range normalisé)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean() / close  # normalisé par le prix

    # Garman-Klass volatility estimator (plus précis qu'un simple std)
    # σ²_GK = 0.5*ln(H/L)² - (2ln2-1)*ln(C/O)²
    log_hl = np.log(high / low)
    log_co = np.log(close / df_1m["open"])
    out["vol_gk"] = (0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2).rolling(15).mean()

    return out


# =============================================================================
# C. MOMENTUM & AUTOCORRELATION
# =============================================================================

def compute_momentum(close: pd.Series) -> pd.DataFrame:
    """
    ROC multi-échelle + autocorrélation des returns.
    """
    out = pd.DataFrame(index=close.index)
    ret = np.log(close / close.shift(1))

    # Rate of Change
    for p in [3, 5, 10, 15, 30]:
        out[f"roc_{p}m"] = (close - close.shift(p)) / close.shift(p)

    # RSI 14 (version log-returns)
    delta = ret.copy()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-10)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # Autocorrélation lag 1 et lag 5 sur fenêtre 15m
    # > 0 = momentum, < 0 = mean-reversion
    out["autocorr_1"] = ret.rolling(15).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False
    )
    out["autocorr_5"] = ret.rolling(20).apply(
        lambda x: pd.Series(x).autocorr(lag=5), raw=False
    )

    return out


# =============================================================================
# D. VOLUME & ORDER FLOW
# =============================================================================

def compute_volume_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Volume relatif, VWAP deviation, order flow imbalance estimé.
    """
    close  = df_1m["close"]
    volume = df_1m["volume"]
    high   = df_1m["high"]
    low    = df_1m["low"]
    out    = pd.DataFrame(index=df_1m.index)

    # Volume relatif (z-score sur 30 bougies)
    vol_mean = volume.rolling(30).mean()
    vol_std  = volume.rolling(30).std()
    out["vol_zscore"] = (volume - vol_mean) / (vol_std + 1e-10)

    # Volume ratio court/long
    out["vol_ratio_5_15"] = volume.rolling(5).mean() / (volume.rolling(15).mean() + 1e-10)

    # VWAP sur 15m et déviation du prix actuel
    typical_price = (high + low + close) / 3
    vwap_15 = (typical_price * volume).rolling(15).sum() / volume.rolling(15).sum()
    out["vwap_dev_15m"] = (close - vwap_15) / (vwap_15 + 1e-10)

    # Order Flow Imbalance proxy (Easley et al.)
    # Heuristique : si close > open → volume "buy", sinon "sell"
    buy_vol  = np.where(close >= df_1m["open"], volume, 0)
    sell_vol = np.where(close <  df_1m["open"], volume, 0)
    buy_roll  = pd.Series(buy_vol,  index=df_1m.index).rolling(15).sum()
    sell_roll = pd.Series(sell_vol, index=df_1m.index).rolling(15).sum()
    total     = buy_roll + sell_roll
    out["ofi_15m"] = (buy_roll - sell_roll) / (total + 1e-10)

    # Même chose sur 5m
    buy_roll5  = pd.Series(buy_vol,  index=df_1m.index).rolling(5).sum()
    sell_roll5 = pd.Series(sell_vol, index=df_1m.index).rolling(5).sum()
    total5     = buy_roll5 + sell_roll5
    out["ofi_5m"] = (buy_roll5 - sell_roll5) / (total5 + 1e-10)

    return out


# =============================================================================
# E. ENTROPY (Lopez de Prado, Ch.18)
# =============================================================================

def compute_entropy(ret: pd.Series, window: int = 15, bins: int = 10) -> pd.Series:
    """
    Shannon entropy des returns sur une fenêtre glissante.

    Haute entropie → returns désordonnés → marché incertain
    Basse entropie → returns structurés → signal plus fiable

    On discrétise les returns en `bins` intervalles puis on calcule H(X).
    """
    def _entropy(x):
        counts, _ = np.histogram(x, bins=bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))

    result = ret.rolling(window).apply(_entropy, raw=True)
    return result.rename("entropy_15m")


# =============================================================================
# F. FEATURES MULTI-TIMEFRAME (5m et 15m)
# =============================================================================

def compute_multitf_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Resample en 5m et calcule des features agrégées.
    Donne le contexte "macro" au modèle.
    """
    df = df_1m.set_index("timestamp").sort_index()

    # Resample 5m
    df_5m = df.resample("5min", closed="left", label="left").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()

    ret_5m = np.log(df_5m["close"] / df_5m["close"].shift(1))

    features_5m = pd.DataFrame(index=df_5m.index)
    features_5m["ret_5m_bar"]   = ret_5m
    features_5m["vol_5m_bar"]   = ret_5m.rolling(3).std()
    features_5m["roc_3bar_5m"]  = (df_5m["close"] - df_5m["close"].shift(3)) / df_5m["close"].shift(3)
    features_5m["roc_6bar_5m"]  = (df_5m["close"] - df_5m["close"].shift(6)) / df_5m["close"].shift(6)

    # Resample 15m
    df_15m = df.resample("15min", closed="left", label="left").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()

    ret_15m = np.log(df_15m["close"] / df_15m["close"].shift(1))

    features_15m = pd.DataFrame(index=df_15m.index)
    features_15m["ret_15m_bar"]  = ret_15m
    features_15m["ret_15m_bar2"] = np.log(df_15m["close"] / df_15m["close"].shift(2))
    features_15m["vol_15m_bar"]  = ret_15m.rolling(4).std()
    features_15m["high_low_15m"] = (df_15m["high"] - df_15m["low"]) / df_15m["close"]

    # Forward-fill sur l'index 1m pour merger ensuite
    features_5m_ff  = features_5m.reindex(df.index, method="ffill")
    features_15m_ff = features_15m.reindex(df.index, method="ffill")

    features_5m_ff.columns  = [f"tf5_{c}"  for c in features_5m_ff.columns]
    features_15m_ff.columns = [f"tf15_{c}" for c in features_15m_ff.columns]

    out = pd.concat([features_5m_ff, features_15m_ff], axis=1)
    out.index.name = "timestamp"
    return out


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def build_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline complet : OHLCV 1m → DataFrame de features.

    Args:
        df_1m : DataFrame avec colonnes [timestamp, open, high, low, close, volume]
                timestamp doit être tz-aware (UTC).

    Returns:
        DataFrame avec toutes les features, index = timestamp 1m.
        Les NaN des premières lignes (warm-up) sont conservés —
        on les droppera au moment de l'entraînement.
    """
    df = df_1m.copy().sort_values("timestamp").reset_index(drop=True)

    close  = df["close"]
    volume = df["volume"]
    ret    = np.log(close / close.shift(1))

    logger.info(f"Build features sur {len(df)} bougies 1m...")

    # --- A. Returns ---
    feat_ret = compute_returns(close)

    # --- A2. Fractional diff (d=0.4 — valeur LdP recommandée) ---
    frac = fractional_diff(close, d=0.4)
    frac_30 = fractional_diff(close, d=0.3)

    # --- B. Volatilité ---
    feat_vol = compute_volatility(df)

    # --- C. Momentum ---
    feat_mom = compute_momentum(close)

    # --- D. Volume ---
    feat_volume = compute_volume_features(df)

    # --- E. Entropy ---
    feat_entropy = compute_entropy(ret, window=15)

    # --- F. Multi-TF ---
    feat_mtf = compute_multitf_features(df)


    # --- Assemblage final ---
    # On utilise df["timestamp"] comme index de référence et on assigne colonne par colonne
    # pour éviter tout doublon de lignes lors du concat multi-index
    base = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    base = base.set_index("timestamp")

    # Tous les DataFrames partagent le même RangeIndex → concat safe
    dfs_to_concat = [
        base.reset_index(drop=True),
        feat_ret.reset_index(drop=True),
        frac.rename("frac_diff_04").reset_index(drop=True),
        frac_30.rename("frac_diff_03").reset_index(drop=True),
        feat_vol.reset_index(drop=True),
        feat_mom.reset_index(drop=True),
        feat_volume.reset_index(drop=True),
        feat_entropy.reset_index(drop=True),
        feat_mtf.reset_index(drop=True),
    ]
    result = pd.concat(dfs_to_concat, axis=1)
    result.insert(0, "timestamp", df["timestamp"].values)

    n_features = result.shape[1] - 6   # hors OHLCV
    logger.info(f"Features construites : {n_features} features | shape={result.shape}")

    return result.reset_index()


# =============================================================================
# MODE LIVE — une seule ligne de features pour le bot
# =============================================================================

def build_live_features(df_1m: pd.DataFrame) -> Optional[pd.Series]:
    """
    Construit les features pour le moment présent (dernière ligne).
    Utilisé par le bot live à chaque cycle de 15 minutes.

    Args:
        df_1m : 60+ dernières bougies 1m (fetch_live depuis binance_feed)

    Returns:
        pd.Series avec toutes les features de la dernière bougie complète,
        ou None si données insuffisantes.
    """
    MIN_BARS = 35   # minimum pour que toutes les fenêtres soient valides

    if len(df_1m) < MIN_BARS:
        logger.error(f"Données insuffisantes : {len(df_1m)} bougies (min={MIN_BARS})")
        return None

    feat_df = build_features(df_1m)

    # Dernière ligne (bougie la plus récente clôturée)
    last = feat_df.iloc[-1]

    # Colonnes features seulement (hors OHLCV brut)
    feature_cols = [c for c in feat_df.columns
                    if c not in ("timestamp", "open", "high", "low", "close", "volume")]

    live_features = last[feature_cols]

    n_nan = live_features.isna().sum()
    if n_nan > 0:
        logger.warning(f"{n_nan} features NaN en live — warm-up insuffisant ?")

    return live_features


# =============================================================================
# CLI — test rapide
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Génère des données synthétiques pour tester sans Binance
    np.random.seed(42)
    n = 500
    price = 100_000 * np.cumprod(1 + np.random.normal(0, 0.001, n))
    timestamps = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")

    df_test = pd.DataFrame({
        "timestamp": timestamps,
        "open"     : price * (1 + np.random.normal(0, 0.0002, n)),
        "high"     : price * (1 + np.abs(np.random.normal(0, 0.0005, n))),
        "low"      : price * (1 - np.abs(np.random.normal(0, 0.0005, n))),
        "close"    : price,
        "volume"   : np.random.lognormal(10, 1, n),
    })

    print("=== Test build_features ===")
    feat = build_features(df_test)
    print(f"Shape : {feat.shape}")
    print(f"Colonnes ({len(feat.columns)}) :")
    for col in feat.columns:
        print(f"  {col}")

    print("\n=== Test build_live_features ===")
    live = build_live_features(df_test.tail(60).reset_index(drop=True))
    if live is not None:
        print(f"Features live : {len(live)} valeurs")
        print(live.dropna().head(10))