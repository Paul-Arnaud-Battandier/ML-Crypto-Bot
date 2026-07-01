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




# === Paths ===

from pathlib import Path

ROOT_DIR = Path(__file__).parent

PATHS = {
    # Regime Detector
    'regime_json'        : ROOT_DIR / 'Regime_Detector'    / 'data' / 'current_regime.json',
    'best_pair_json'      : ROOT_DIR / 'Regime_Detector'    / 'data' / 'best_pair.json',
    'regime_scripts'      : ROOT_DIR / 'Regime_Detector'    / 'scripts',

    # StatArb
    'statarb_data'        : ROOT_DIR / 'StatArb_ETH_15m'    / 'data',
    'statarb_model'        : ROOT_DIR / 'StatArb_ETH_15m'    / 'model' / 'lgbm_statarb.pkl',
    'live_equity'          : ROOT_DIR / 'StatArb_ETH_15m'    / 'data'  / 'live_equity.csv',
    'live_trades'          : ROOT_DIR / 'StatArb_ETH_15m'    / 'data'  / 'live_trades.csv',
    'statarb_scripts'      : ROOT_DIR / 'StatArb_ETH_15m'    / 'scripts',

    # Funding Rate Carry
    'funding_data'         : ROOT_DIR / 'FundingCarry_Multi' / 'data',
    'best_funding_json'    : ROOT_DIR / 'FundingCarry_Multi' / 'data' / 'best_funding.json',
    'funding_state_json'   : ROOT_DIR / 'FundingCarry_Multi' / 'data' / 'funding_state.json',
    'funding_equity'       : ROOT_DIR / 'FundingCarry_Multi' / 'data' / 'funding_equity.csv',
    'funding_trades'       : ROOT_DIR / 'FundingCarry_Multi' / 'data' / 'funding_trades.csv',
    'funding_scripts'      : ROOT_DIR / 'FundingCarry_Multi' / 'scripts',

    # Web Dashboard
    'dashboard_static'     : ROOT_DIR / 'Web_Dashboard'      / 'static',
    'dashboard_templates'  : ROOT_DIR / 'Web_Dashboard'      / 'templates',
}