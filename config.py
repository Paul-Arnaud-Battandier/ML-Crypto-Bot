# === Data parameters ===
SYMBOL      = "BTC/USDT"
SYMBOL_FUTURES = "BTCUSDT"
TIMEFRAME   = "4h"
START_DATE  = "2020-01-01"
END_DATE    = "2024-01-01"
HOLDOUT_END = "2024-07-01"

# === Paths ===
DATA_RAW       = "data/raw/"
DATA_PROCESSED = "data/processed/"
MODELS_DIR     = "models/"

# === Binance ===
BINANCE_FUTURES_URL = "https://fapi.binance.com/fapi/v1/fundingRate"




"""
config.py
─────────
Chemins centralisés pour tous les modules du projet.
Importer avec : from config import PATHS
"""
from pathlib import Path

ROOT_DIR = Path(__file__).parent

PATHS = {
    # Regime Detector
    'regime_json'   : ROOT_DIR / 'Regime_Detector'  / 'data' / 'current_regime.json',
    'best_pair_json': ROOT_DIR / 'Regime_Detector'  / 'data' / 'best_pair.json',

    # StatArb
    'statarb_data'  : ROOT_DIR / 'StatArb_ETH_15m'  / 'data',
    'statarb_model' : ROOT_DIR / 'StatArb_ETH_15m'  / 'model' / 'lgbm_statarb.pkl',
    'live_equity'   : ROOT_DIR / 'StatArb_ETH_15m'  / 'data'  / 'live_equity.csv',
    'live_trades'   : ROOT_DIR / 'StatArb_ETH_15m'  / 'data'  / 'live_trades.csv',
}