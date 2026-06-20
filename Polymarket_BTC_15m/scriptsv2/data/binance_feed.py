"""
binance_feed.py
---------------
Collecte OHLCV 1m BTC/USDT depuis Binance via ccxt.

Deux modes :
  - backfill(days=90)  → remplit data/binance_1m_raw.csv depuis N jours
  - fetch_live()       → retourne les 60 dernières bougies 1m (DataFrame)

Pas de clé API nécessaire pour les données publiques.
"""

import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import ccxt
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SYMBOL      = "BTC/USDT"
TIMEFRAME   = "1m"
LIMIT       = 1000          # max bougies par requête Binance
LIVE_BARS   = 60            # bougies retournées en mode live (1h de contexte)
RAW_CSV     = Path(__file__).parents[2] / "data" / "binance_1m_raw.csv"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Initialisation exchange (public, pas de clé)
# ---------------------------------------------------------------------------
def _get_exchange() -> ccxt.binance:
    return ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })


# ---------------------------------------------------------------------------
# Fetch générique : une plage de timestamps
# ---------------------------------------------------------------------------
def _fetch_range(
    exchange: ccxt.binance,
    since_ms: int,
    until_ms: int,
) -> pd.DataFrame:
    """Récupère tous les OHLCV 1m entre since_ms et until_ms (ms epoch)."""
    all_rows = []
    cursor   = since_ms

    while cursor < until_ms:
        try:
            raw = exchange.fetch_ohlcv(
                SYMBOL, TIMEFRAME, since=cursor, limit=LIMIT
            )
        except ccxt.NetworkError as e:
            logger.warning(f"NetworkError, retry in 5s : {e}")
            time.sleep(5)
            continue
        except ccxt.ExchangeError as e:
            logger.error(f"ExchangeError : {e}")
            raise

        if not raw:
            break

        all_rows.extend(raw)
        cursor = raw[-1][0] + 60_000   # +1 minute en ms
        logger.debug(f"Fetched up to {pd.Timestamp(cursor, unit='ms', tz='UTC')}")
        time.sleep(exchange.rateLimit / 1000)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[df["timestamp"] < pd.Timestamp(until_ms, unit="ms", tz="UTC")]
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Mode 1 : backfill historique → CSV
# ---------------------------------------------------------------------------
def backfill(days: int = 90, force: bool = False) -> pd.DataFrame:
    """
    Télécharge `days` jours de bougies 1m et les sauvegarde dans RAW_CSV.
    Si le CSV existe déjà, ne télécharge que les données manquantes (sauf force=True).

    Returns:
        DataFrame complet après mise à jour.
    """
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    exchange = _get_exchange()

    now_ms    = int(datetime.now(timezone.utc).timestamp() * 1000)
    target_ms = now_ms - days * 24 * 3600 * 1000

    # --- Chargement du CSV existant ---
    if RAW_CSV.exists() and not force:
        existing = pd.read_csv(RAW_CSV, parse_dates=["timestamp"])
        existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
        last_ts_ms = int(existing["timestamp"].max().timestamp() * 1000)
        since_ms   = last_ts_ms + 60_000
        logger.info(
            f"CSV existant : {len(existing)} bougies. "
            f"Reprise depuis {pd.Timestamp(since_ms, unit='ms', tz='UTC')}"
        )
    else:
        existing  = pd.DataFrame()
        since_ms  = target_ms
        logger.info(f"Nouveau backfill : {days} jours depuis "
                    f"{pd.Timestamp(since_ms, unit='ms', tz='UTC')}")

    # --- Fetch nouvelles données ---
    new_data = _fetch_range(exchange, since_ms, now_ms)

    if new_data.empty:
        logger.info("Aucune nouvelle bougie à télécharger.")
        return existing if not existing.empty else pd.DataFrame()

    # --- Fusion et sauvegarde ---
    if not existing.empty:
        df = pd.concat([existing, new_data], ignore_index=True)
    else:
        df = new_data

    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df.to_csv(RAW_CSV, index=False)
    logger.info(f"CSV sauvegardé : {len(df)} bougies → {RAW_CSV}")
    return df


# ---------------------------------------------------------------------------
# Mode 2 : fetch live → DataFrame (utilisé par le bot)
# ---------------------------------------------------------------------------
def fetch_live(n_bars: int = LIVE_BARS) -> pd.DataFrame:
    """
    Retourne les `n_bars` dernières bougies 1m complètes.
    La bougie en cours (non clôturée) est exclue.

    Returns:
        DataFrame avec colonnes [timestamp, open, high, low, close, volume]
    """
    exchange = _get_exchange()
    now_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    since_ms = now_ms - (n_bars + 2) * 60_000   # +2 pour margin

    df = _fetch_range(exchange, since_ms, now_ms)

    if df.empty:
        logger.error("fetch_live : aucune donnée reçue")
        return df

    # Exclure la bougie en cours (non clôturée)
    current_minute = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    df = df[df["timestamp"] < pd.Timestamp(current_minute)]

    return df.tail(n_bars).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Utilitaire : resample 1m → 5m et 15m
# ---------------------------------------------------------------------------
def resample(df: pd.DataFrame, rule: str = "5min") -> pd.DataFrame:
    """
    Resample un DataFrame OHLCV 1m vers une timeframe supérieure.

    Args:
        df   : DataFrame avec colonne 'timestamp' (tz-aware) et OHLCV
        rule : '5min', '15min', '1h', etc.

    Returns:
        DataFrame resampleé, index reset.
    """
    df = df.set_index("timestamp").sort_index()
    resampled = df.resample(rule, closed="left", label="left").agg({
        "open"  : "first",
        "high"  : "max",
        "low"   : "min",
        "close" : "last",
        "volume": "sum",
    }).dropna()
    return resampled.reset_index()


# ---------------------------------------------------------------------------
# CLI rapide : python -m scriptsv2.data.binance_feed
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger.info("=== Backfill Binance 1m — 90 jours ===")
    df = backfill(days=90)
    print(df.tail())
    print(f"\nTotal : {len(df)} bougies")
    print(f"Période : {df['timestamp'].min()} → {df['timestamp'].max()}")