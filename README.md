# ML Crypto Bot — Regime-Switching Market-Neutral Trading System

A quantitative research project exploring systematic, market-neutral trading strategies on crypto perpetuals, orchestrated by a macro regime detector. Built end-to-end: data pipelines, statistical validation, ML meta-labeling, live paper-trading deployment, and a public real-time dashboard.

**Live dashboard:** [algo-trading-bot-d6yk.onrender.com](https://algo-trading-bot-d6yk.onrender.com)
*(Paper trading only — Binance Demo Trading, no real funds involved)*

---

## Architecture

```
                    ┌─────────────────────┐
                    │   Regime Detector    │
                    │  (BTC vol + ETH      │
                    │   Hurst/ADX, hourly) │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                 │
        MEAN_REV          TRENDING           HIGH_VOL/NEUTRAL
              │                │                 │
              ▼                ▼                 ▼
      ┌───────────────┐  ┌───────────┐  ┌──────────────────┐
      │ Statistical    │  │ (unallocated)│ │ Funding Rate     │
      │ Arbitrage      │  │ Momentum   │  │ Carry            │
      │ AAVE/ETH       │  │ rejected — │  │ Long Spot +      │
      │ + LightGBM     │  │ see below  │  │ Short Perpetual  │
      └───────────────┘  └───────────┘  └──────────────────┘
              │                                  │
              └────────────────┬─────────────────┘
                                ▼
                        Supabase (state)
                                ▼
                     Flask Dashboard (Render)
```

Every component runs autonomously as a background thread inside a single Render web service, writing state to Supabase (PostgreSQL) for persistence across restarts and for the live dashboard to read from.

---

## Strategies

### 🟢 Layer 0 — Regime Detector
A two-level macro filter recomputed hourly:
- **BTC/USDT realized volatility** detects systemic stress → triggers `HIGH_VOL`.
- **ETH/USDT Hurst Exponent + ADX** detects the microstructure regime → `MEAN_REV` or `TRENDING`.

Validated via walk-forward testing across three distinct 4-month market periods (including a -38% BTC drawdown), with a consistent Hurst separation of 0.21–0.23 between regimes — confirming the classifier generalizes across market direction, not just the period it was calibrated on.

### 📊 Layer 1 — Statistical Arbitrage (active in `MEAN_REV`)
Pairs trading on AAVE/USDT vs ETH/USDT:
- Pair selected via Engle-Granger cointegration + Hurst Exponent + out-of-sample ADF stationarity.
- **Dynamic hedge ratio** via Rolling OLS (institutional standard — avoids stale-ratio drift).
- **LightGBM meta-labeling** gatekeeper trained on Lopez de Prado's Triple Barrier Method, raising the raw engine's 58.7% win-rate to 88.7% out-of-sample.
- Three independent stop-loss layers: Z-Score (|Z|≥6), PnL (-5%), Time Barrier (48h).
- Weekly automated re-scan across 11 candidate pairs to adapt to structural market changes.

### 💰 Layer 2 — Funding Rate Carry (active in `HIGH_VOL` / `NEUTRAL`)
Delta-neutral carry: **Long Spot + Short Perpetual**, harvesting the funding rate paid every 8h by leveraged longs — zero directional exposure.
- 20-symbol universe scanned daily; only pairs clearing a strict post-fee breakeven filter (30d APR > 4%, positive-payment consistency > 60%) are traded.
- Exit logic uses a **rolling average of the last 5 payments** (not a single data point) to avoid over-reacting to one negative funding cycle.
- Fee/breakeven math computed *before* deployment — at scan time, only 1 of 20 symbols cleared the bar, confirming the filter isn't cosmetic.

### 🔬 Rejected — Cross-Sectional Momentum (research only, not deployed)
Backtested to fill the `TRENDING` regime slot. Result: Sharpe -1.28, no grid-search configuration produced a robust edge — the return-ranking signal was effectively inverted during the test window (mean-reverting macro backdrop). Documented as a **deliberate negative result**: the `TRENDING` slot remains intentionally unallocated rather than deployed with an unvalidated edge.

### ❌ Abandoned earlier phases (see `research.txt`)
- **BTC/USDT 4H directional** (XGBoost + Random Forest meta-labeling): unprofitable after Triple Barrier backtest — pivoted.
- **Polymarket 15m binary contracts**: mathematically unviable due to $0.98 bid-ask spread on target markets (requires >99.1% win-rate to break even).

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data & Execution | `ccxt`, Binance Demo Trading API (Spot + Futures) |
| Statistics | `statsmodels` (cointegration, ADF, OLS), custom Hurst/ADX implementations |
| Machine Learning | `LightGBM`, `scikit-learn` |
| Backend | `Flask`, `gunicorn`, background threads (regime / statarb / funding loops) |
| Persistence | `Supabase` (PostgreSQL, REST API) |
| Hosting | `Render` (free tier, single web service) |
| Frontend | Vanilla HTML/CSS/JS, `Chart.js` |

---

## Repository Structure

```
ML_Crypto_Bot/
├── Regime_Detector/          # Macro regime classification (BTC vol + ETH Hurst/ADX)
├── StatArb_ETH_15m/          # Pairs trading: cointegration scan → ML → live bot
├── FundingCarry_Multi/       # Delta-neutral funding rate harvesting
├── Momentum_CrossSectional/  # Rejected strategy, kept for research transparency
├── BTC_4h/                   # Abandoned Phase 1 (directional swing trading)
├── Polymarket_BTC_15m/       # Abandoned Phase 2 (binary prediction markets)
├── Web_Dashboard/            # Flask app + background job orchestration
├── config.py                 # Centralized path configuration
└── research.txt              # Full research log, phase by phase
```

Each strategy folder follows the same pattern: `research/` (notebooks, exploratory scanners) → `scripts/` (production pipeline: fetch → features → train → backtest → live) → `data/` & `model/` (gitignored, regenerated locally).

---

## Running Locally

```bash
# Environment
cp .env.example .env   # fill in API_KEY, API_SECRET, SUPABASE_URL, SUPABASE_KEY

# Regime detector (background, hourly)
python Regime_Detector/scripts/live_regime.py

# Statistical Arbitrage bot
python StatArb_ETH_15m/scripts/live_bot.py

# Funding Rate Carry bot
python FundingCarry_Multi/scripts/live_funding_bot.py

# Dashboard (local)
cd Web_Dashboard && python app.py
```

On Render, all three bots run as daemon threads inside the single Flask process (`RUN_BACKGROUND_JOBS=true`), staggered on startup to respect exchange rate limits.

---

## Disclaimer

This project trades exclusively on **Binance Demo Trading** (paper money, simulated fills). No real capital is at risk. Nothing here constitutes financial advice. All performance figures are from paper trading and should be read as research validation, not live track record claims.

---

## About

Paul Arnaud-Battandier — Finance & Quantitative Engineering, ECE.
Seeking a 6 month end-of-studies internship (Jan 2027) in trading or quantitative research.
[LinkedIn](https://www.linkedin.com/in/paul-arnaud-battandier/) · [GitHub](https://github.com/Paul-Arnaud-Battandier) · [CV](Web_Dashboard/static/CV_Paul_Arnaud-Battandier.pdf)
