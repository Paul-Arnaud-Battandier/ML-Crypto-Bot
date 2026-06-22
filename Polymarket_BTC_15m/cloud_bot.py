"""
cloud_bot.py
------------
Entrypoint principal pour Render.

Architecture :
  - Flask          → keep-alive + endpoints /csv /status /health
  - APScheduler    → déclenche le bot toutes les 15 minutes pile
  - Thread daemon  → le bot tourne en arrière-plan sans bloquer Flask

Endpoints :
  /           → statut rapide (keep-alive ping)
  /health     → JSON statut détaillé
  /csv        → historique des trades en HTML
  /status     → dernière décision du bot en JSON

Déploiement Render :
  - Procfile : web: gunicorn cloud_bot:app
  - Le bot se lance automatiquement au démarrage de Gunicorn
  - UptimeRobot pinge / toutes les 5 min pour éviter le sleep
"""

import csv
import json
import logging
import os
import sys
import threading
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

from Polymarket_BTC_15m.scriptsv2.data.polymarket_feed import get_current_15m_slug

# --- Ajout du répertoire racine au path ---
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))  # ajoute ML_Crypto_Bot aussi

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from flask import Flask, jsonify, Response

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("cloud_bot")

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
DATA_DIR    = ROOT / "data"
MODEL_DIR   = ROOT / "model"
TRADE_LOG   = DATA_DIR / "trade_history.csv"
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# État global du bot (thread-safe via lock)
# ---------------------------------------------------------------------------
_state_lock = threading.Lock()
_bot_state  = {
    "last_run"      : None,
    "last_decision" : None,
    "n_runs"        : 0,
    "n_trades"      : 0,
    "n_skips"       : 0,
    "errors"        : [],
    "started_at"    : datetime.now(timezone.utc).isoformat(),
}


def _update_state(**kwargs):
    with _state_lock:
        _bot_state.update(kwargs)


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------
CSV_HEADERS = [
    "timestamp", "window_start", "slug",
    "lgbm_proba", "meta_proba", "direction", "side",
    "trade", "reason",
    "poly_yes_mid", "poly_no_mid", "poly_spread",
    "btc_close_entry",
]


def log_trade(record: dict):
    """Ajoute une ligne au CSV de trade history."""
    TRADE_LOG.parent.mkdir(exist_ok=True)
    write_header = not TRADE_LOG.exists()

    with open(TRADE_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(record)

    logger.info(f"Trade loggé → {record.get('side', 'SKIP')} | "
                f"LGBM={record.get('lgbm_proba', 'N/A')} | "
                f"Meta={record.get('meta_proba', 'N/A')}")


# ---------------------------------------------------------------------------
# CYCLE BOT — exécuté toutes les 15 minutes
# ---------------------------------------------------------------------------

def run_cycle():
    """
    Un cycle complet du bot :
      1. Vérification de la fenêtre temporelle (pas trop tôt, pas trop tard)
      2. Fetch données Binance live
      3. Calcul features
      4. Prédiction LGBM
      5. Confirmation meta RF + Polymarket
      6. Log de la décision
    """
    now = datetime.now(timezone.utc)
    logger.info(f"{'='*60}")
    logger.info(f"CYCLE BOT | {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    try:
        # ------------------------------------------------------------------
        # 0. Garde-fou temporel
        # ------------------------------------------------------------------
        elapsed_s = (now.minute % 15) * 60 + now.second

        if elapsed_s < 30:
            logger.info(f"Trop tôt dans la tranche ({elapsed_s}s) — skip")
            _update_state(last_run=now.isoformat(), n_skips=_bot_state["n_skips"] + 1)
            return

        if elapsed_s > 13 * 60:
            logger.info(f"Trop tard dans la tranche ({elapsed_s}s) — skip")
            _update_state(last_run=now.isoformat(), n_skips=_bot_state["n_skips"] + 1)
            return

        # ------------------------------------------------------------------
        # 1. Données Binance live
        # ------------------------------------------------------------------
        from scriptsv2.data.binance_feed import fetch_live
        df_live = fetch_live(n_bars=90)

        if df_live is None or len(df_live) < 35:
            logger.error(f"Données Binance insuffisantes : {len(df_live) if df_live is not None else 0} bougies")
            return

        btc_close = float(df_live["close"].iloc[-1])
        logger.info(f"BTC close : {btc_close:,.2f} USDT | {len(df_live)} bougies 1m")

        # ------------------------------------------------------------------
        # 2. Features
        # ------------------------------------------------------------------
        from scriptsv2.features.feature_pipeline import build_live_features
        live_features = build_live_features(df_live)

        if live_features is None:
            logger.error("Impossible de construire les features live")
            return

        # ------------------------------------------------------------------
        # 3. Chargement modèles
        # ------------------------------------------------------------------
        from scriptsv2.training.train_lgbm import load_model
        from scriptsv2.training.train_meta import load_meta_model

        lgbm_payload = load_model(MODEL_DIR / "lgbm_model.pkl")
        meta_payload = load_meta_model(MODEL_DIR / "meta_model.pkl")

        if lgbm_payload is None or meta_payload is None:
            logger.error("Modèles non disponibles — lance train_lgbm.py et train_meta.py")
            return

        # ------------------------------------------------------------------
        # 4. Snapshot Polymarket
        # ------------------------------------------------------------------
        poly_snapshot = None
        try:
            from scriptsv2.data.polymarket_feed import get_market_snapshot
            poly_snapshot = get_market_snapshot()
            if poly_snapshot:
                logger.info(f"Polymarket | YES={poly_snapshot.get('yes_mid', 'N/A')} | "
                            f"NO={poly_snapshot.get('no_mid', 'N/A')}")
        except Exception as e:
            logger.warning(f"Polymarket indisponible : {e} — on continue sans")

        # ------------------------------------------------------------------
        # 5. Décision combinée
        # ------------------------------------------------------------------
        from scriptsv2.training.train_meta import predict_combined

        decision = predict_combined(
            binance_features=live_features,
            poly_snapshot=poly_snapshot,
            lgbm_payload=lgbm_payload,
            meta_payload=meta_payload,
            min_lgbm_conf=0.55,
            min_meta_prob=0.55,
        )

        logger.info(f"DÉCISION : {'✅ TRADE' if decision['trade'] else '⏭ SKIP'} | "
                    f"{decision['direction']} | {decision['reason']}")

        # ------------------------------------------------------------------
        # 6. Logging CSV
        # ------------------------------------------------------------------
        from scriptsv2.data.polymarket_feed import get_current_15m_slug
        slug, window_start = get_current_15m_slug()

        record = {
            "timestamp"       : now.isoformat(),
            "window_start"    : window_start.isoformat(),
            "slug"            : slug,
            "lgbm_proba"      : decision.get("lgbm_proba"),
            "meta_proba"      : decision.get("meta_proba"),
            "direction"       : decision.get("direction"),
            "side"            : decision.get("side"),
            "trade"           : decision.get("trade"),
            "reason"          : decision.get("reason"),
            "poly_yes_mid"    : poly_snapshot.get("yes_mid")    if poly_snapshot else None,
            "poly_no_mid"     : poly_snapshot.get("no_mid")     if poly_snapshot else None,
            "poly_spread"     : poly_snapshot.get("yes_spread") if poly_snapshot else None,
            "btc_close_entry" : btc_close,
        }
        log_trade(record)

        # Mise à jour état global
        _update_state(
            last_run=now.isoformat(),
            last_decision=decision,
            n_runs=_bot_state["n_runs"] + 1,
            n_trades=_bot_state["n_trades"] + (1 if decision["trade"] else 0),
            n_skips=_bot_state["n_skips"] + (0 if decision["trade"] else 1),
        )

    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        logger.error(f"Erreur cycle bot : {err_msg}")
        logger.error(traceback.format_exc())
        with _state_lock:
            _bot_state["errors"].append({
                "time" : now.isoformat(),
                "error": err_msg,
            })
            # Garder seulement les 10 dernières erreurs
            _bot_state["errors"] = _bot_state["errors"][-10:]


# ---------------------------------------------------------------------------
# SCHEDULER APScheduler
# ---------------------------------------------------------------------------

def start_scheduler():
    """
    Lance le scheduler APScheduler.
    Déclenche run_cycle() à hh:00, hh:15, hh:30, hh:45 + 2 minutes.

    On décale de 2 minutes (+2) pour :
      - Laisser le temps à Polymarket d'ouvrir le nouveau marché
      - Avoir les bougies Binance complètes
      - Éviter les edge cases de timing
    """
    scheduler = BackgroundScheduler(timezone="UTC")

    # Toutes les 15 minutes à MM:02 (hh:02, hh:17, hh:32, hh:47)
    scheduler.add_job(
        run_cycle,
        trigger=CronTrigger(minute="2,17,32,47", second="0"),
        id="bot_cycle",
        name="BTC 15m bot cycle",
        replace_existing=True,
        misfire_grace_time=60,
    )

    scheduler.start()
    logger.info("Scheduler démarré | Cycles : hh:02, hh:17, hh:32, hh:47 UTC")
    return scheduler


# ---------------------------------------------------------------------------
# HELPERS — lecture CSV
# ---------------------------------------------------------------------------

def read_trades():
    """Lit trade_history.csv et retourne une liste de dicts, plus récent en premier."""
    if not TRADE_LOG.exists():
        return []
    rows = []
    try:
        with open(TRADE_LOG, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        rows.reverse()
    except Exception as e:
        logger.error(f"Erreur lecture CSV : {e}")
    return rows


def compute_stats(trades):
    """Calcule les métriques globales depuis la liste de trades."""
    confirmed = [t for t in trades if t.get("trade") == "True"]
    n_total   = len(trades)
    n_trades  = len(confirmed)
    n_skips   = n_total - n_trades

    # Win = à implémenter quand on a le résultat Polymarket
    # Pour l'instant on expose les métriques disponibles
    lgbm_probas = []
    for t in confirmed:
        try:
            lgbm_probas.append(float(t["lgbm_proba"]))
        except (KeyError, ValueError):
            pass

    avg_conf = sum(lgbm_probas) / len(lgbm_probas) if lgbm_probas else 0

    return {
        "n_total"   : n_total,
        "n_trades"  : n_trades,
        "n_skips"   : n_skips,
        "trade_rate": f"{n_trades/n_total*100:.1f}" if n_total else "0",
        "avg_lgbm"  : f"{avg_conf:.3f}",
        "up_count"  : sum(1 for t in confirmed if t.get("direction") == "UP"),
        "down_count": sum(1 for t in confirmed if t.get("direction") == "DOWN"),
    }


# ---------------------------------------------------------------------------
# FLASK APP
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/ping")
def ping():
    """Endpoint ultra-léger pour UptimeRobot — pas de lecture CSV."""
    return "ok", 200


@app.route("/health")
def health():
    """Statut détaillé en JSON."""
    with _state_lock:
        state = dict(_bot_state)
    return jsonify({
        "status"       : "running",
        "uptime"       : state["started_at"],
        "stats"        : {"n_runs": state["n_runs"], "n_trades": state["n_trades"], "n_skips": state["n_skips"], "last_run": state["last_run"]},
        "last_decision": state["last_decision"],
        "recent_errors": state["errors"][-3:],
    })


@app.route("/api/trades")
def api_trades():
    """API JSON — retourne les trades pour le dashboard JS."""
    trades = read_trades()
    stats  = compute_stats(trades)
    with _state_lock:
        bot_state = dict(_bot_state)
    return jsonify({
        "trades"    : trades[:50],
        "stats"     : stats,
        "bot_state" : {"n_runs": bot_state["n_runs"], "last_run": bot_state["last_run"]},
    })


@app.route("/api/polymarket-current")
def polymarket_current():
    slug, window_start = get_current_15m_slug()
    return jsonify({
        "slug": slug,
        "window_start": window_start.isoformat(),
    })


@app.route("/run_now")
def run_now():
    """Déclenche un cycle manuellement (debug)."""
    thread = threading.Thread(target=run_cycle)
    thread.daemon = True
    thread.start()
    return jsonify({"status": "cycle lancé"})


@app.route("/")
def dashboard():
    """Dashboard principal — 3 onglets, données live depuis /api/trades."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BTC/Polymarket — Algo Trading Bot</title>
<meta http-equiv="refresh" content="120">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.31.0/dist/tabler-icons.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f1117;color:#e2e8f0;min-height:100vh}
.container{max-width:1100px;margin:0 auto;padding:2rem 1.5rem}
.header{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:2rem;padding-bottom:1.5rem;border-bottom:1px solid #1e2432}
.header-left h1{font-size:20px;font-weight:500;color:#f1f5f9}
.header-left p{font-size:13px;color:#64748b;margin-top:4px}
.badge-live{display:inline-flex;align-items:center;gap:5px;background:#0f2922;color:#34d399;font-size:11px;padding:4px 10px;border-radius:20px;border:1px solid #065f46}
.badge-live::before{content:'';width:6px;height:6px;border-radius:50%;background:#34d399;animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.tabs{display:flex;gap:0;border-bottom:1px solid #1e2432;margin-bottom:2rem}
.tab{padding:10px 20px;font-size:13px;font-weight:500;color:#64748b;cursor:pointer;border:none;background:none;border-bottom:2px solid transparent;margin-bottom:-1px}
.tab:hover{color:#94a3b8}
.tab.active{color:#f1f5f9;border-bottom:2px solid #3b82f6}
.tab-content{display:none}.tab-content.active{display:block}
.grid-4{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:1.5rem}
.grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:1.5rem}
.grid-2{display:grid;grid-template-columns:repeat(2,1fr);gap:1.25rem;margin-bottom:1.25rem}
.metric{background:#141820;border:1px solid #1e2432;border-radius:10px;padding:1rem}
.metric-label{font-size:11px;color:#475569;margin-bottom:8px;text-transform:uppercase;letter-spacing:.06em}
.metric-value{font-size:24px;font-weight:500;color:#f1f5f9}
.metric-sub{font-size:11px;color:#334155;margin-top:4px}
.card{background:#141820;border:1px solid #1e2432;border-radius:12px;padding:1.25rem;margin-bottom:1.25rem}
.card-title{font-size:11px;font-weight:500;color:#475569;text-transform:uppercase;letter-spacing:.06em;margin-bottom:1rem}
.up{color:#34d399}.down{color:#f87171}
.neutral{color:#64748b}
.pill{display:inline-block;font-size:11px;padding:2px 9px;border-radius:20px;font-weight:500}
.pill-up{background:#052e16;color:#34d399;border:1px solid #065f46}
.pill-down{background:#2d0a0a;color:#f87171;border:1px solid #7f1d1d}
.pill-skip{background:#1e2432;color:#64748b;border:1px solid #2d3748}
table{width:100%;font-size:12px;border-collapse:collapse;table-layout:fixed}
th{font-size:11px;color:#475569;font-weight:500;padding:7px 10px;text-align:left;border-bottom:1px solid #1e2432}
td{padding:8px 10px;border-bottom:1px solid #1a2035;color:#cbd5e1}
tr:last-child td{border-bottom:none}
.step{display:flex;gap:14px;margin-bottom:1.5rem}
.step-num{min-width:30px;height:30px;border-radius:50%;background:#1e3a5f;color:#60a5fa;font-size:12px;font-weight:500;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px}
.step h4{font-size:14px;font-weight:500;color:#f1f5f9;margin-bottom:5px}
.step p{font-size:13px;color:#64748b;line-height:1.65}
.why{border-left:2px solid #1e2432;padding-left:1rem;margin-bottom:1.25rem}
.why.pivot{border-left-color:#3b82f6}
.why h4{font-size:13px;font-weight:500;color:#e2e8f0;margin-bottom:5px}
.why p{font-size:13px;color:#64748b;line-height:1.65}
.tech-tag{display:inline-block;font-size:11px;padding:3px 9px;border-radius:5px;background:#1e2432;color:#64748b;margin:2px;border:1px solid #2d3748}
.chart-wrap{position:relative;width:100%}
.skill-bar{height:3px;background:#1e2432;border-radius:2px;margin-top:5px}
.skill-fill{height:3px;border-radius:2px;background:#3b82f6}
.social-btn{display:inline-flex;align-items:center;gap:7px;padding:8px 16px;border-radius:8px;border:1px solid #2d3748;font-size:13px;color:#e2e8f0;text-decoration:none;margin-right:8px;background:#141820}
.social-btn:hover{background:#1e2432;border-color:#3d4f6b}
.poly-box{border-radius:8px;padding:.85rem 1rem;flex:1;text-align:center}
.poly-yes{background:#052e16;border:1px solid #065f46}
.poly-no{background:#2d0a0a;border:1px solid #7f1d1d}
.poly-price{font-size:26px;font-weight:500}
.poly-lbl{font-size:11px;color:#64748b;margin-top:3px}
.divider{border:none;border-top:1px solid #1e2432;margin:1.5rem 0}
.disclaimer{font-size:11px;color:#334155;padding:.85rem 1rem;background:#0d1117;border-radius:8px;margin-top:1.5rem;border-left:2px solid #1e2432;line-height:1.6}
.profile-avatar{width:68px;height:68px;border-radius:50%;background:#1e3a5f;display:flex;align-items:center;justify-content:center;font-size:20px;font-weight:500;color:#60a5fa;flex-shrink:0}
.feat-row{display:flex;align-items:center;padding:6px 0;border-bottom:1px solid #1a2035;font-size:12px;gap:10px}
.feat-row:last-child{border-bottom:none}
.feat-bar-bg{flex:1;height:3px;background:#1e2432;border-radius:2px}
.feat-bar-fill{height:3px;border-radius:2px}
#last-update{font-size:11px;color:#334155;margin-top:4px}
</style>
</head>
<body>
<div class="container">

<div class="header">
  <div class="header-left">
    <h1>BTC/Polymarket — Algorithmic Trading Bot</h1>
    <p>LightGBM + Random Forest meta-labelling &middot; 15-minute binary prediction &middot; Paper trading</p>
    <p id="last-update">Loading...</p>
  </div>
  <span class="badge-live">Live</span>
</div>

<div class="tabs">
  <button class="tab active" onclick="switchTab('approach',this)">Approach</button>
  <button class="tab" onclick="switchTab('perf',this)">Performance</button>
  <button class="tab" onclick="switchTab('profile',this)">About me</button>
</div>

<!-- ══════════════════════════ TAB 1 — APPROACH -->
<div id="tab-approach" class="tab-content active">
  <div class="card">
    <div class="card-title">From BTC 4H swing trading to Polymarket binary prediction</div>
    <div class="why">
      <h4>Phase 1 — BTC/USDT 4H swing trading (abandoned)</h4>
      <p>Initial strategy: XGBoost with Optuna hyperparameter tuning, funding rates, Fear &amp; Greed index, and macro data (DXY, equities). Full pipeline built: data &rarr; features &rarr; triple-barrier labels &rarr; meta-labelling (Random Forest). After backtesting, the combined model showed <span class="down">no positive expectancy</span> — XGBoost alone: 33.25% win rate, meta-model precision: 55.14% (break-even at 1.5:1 R/R requires &gt;40%). Managing stop-losses, position sizing, and intrabar fluctuations added complexity without edge.</p>
    </div>
    <div class="why pivot">
      <h4>Pivot — why Polymarket 15-minute binary markets?</h4>
      <p>Polymarket's BTC up/down contracts eliminate the core challenge: <em>no stop-loss, no slippage, no intrabar noise</em>. The market resolves exactly at close[t+15min]. Binary label = 1 if close[t+15] &gt; close[t]. This lets the model focus purely on directional quality — measured against a clean, unambiguous ground truth every 15 minutes.</p>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Pipeline — 7 steps</div>
    <div class="step"><div class="step-num">1</div><div><h4>Data collection</h4><p>Binance OHLCV 1m via ccxt (public API). 90-day backfill + incremental live updates. Polymarket CLOB API for real-time bid/ask using a deterministic slug: <code style="font-size:11px;background:#1e2432;padding:1px 6px;border-radius:3px;color:#94a3b8">btc-updown-15m-{unix_ts}</code> — no search query needed, mathematically computed each cycle.</p></div></div>
    <div class="step"><div class="step-num">2</div><div><h4>Feature engineering — Lopez de Prado framework</h4><p>35 features across 6 families: fractional differentiation (d=0.4, preserves long memory while ensuring stationarity), Shannon entropy of returns, Garman-Klass volatility estimator, order flow imbalance, VWAP deviation, autocorrelation, multi-timeframe context (5m + 15m resampled from 1m). Purged K-Fold CV prevents lookahead bias from rolling windows.</p></div></div>
    <div class="step"><div class="step-num">3</div><div><h4>Primary model — LightGBM</h4><p>Trained with Purged K-Fold CV (5 folds, purge=30 bars). OOS AUC: 0.768, ACC: 68.5% — consistent across all folds. 130,978 samples, perfectly balanced (49.7% UP / 50.3% DOWN).</p></div></div>
    <div class="step"><div class="step-num">4</div><div><h4>Meta-labelling — Random Forest</h4><p>The RF predicts whether the LGBM signal is worth trading — not the direction. Meta-label = 1 if LGBM was correct. Filters 64% of signals. OOS precision: 89.3%. Live Polymarket features (spread, implied probability, order imbalance) injected at inference time.</p></div></div>
    <div class="step"><div class="step-num">5</div><div><h4>Combined decision gate</h4><p>Trade only when: LGBM confidence &gt; 55% AND RF meta probability &gt; 55% AND Polymarket spread &lt; 10%. This triple filter ensures capital is only deployed on high-conviction, liquid signals.</p></div></div>
    <div class="step"><div class="step-num">6</div><div><h4>Live bot — Render + APScheduler</h4><p>Deployed on Render (free tier). APScheduler triggers at hh:02, hh:17, hh:32, hh:47 UTC. Flask keeps the service alive; UptimeRobot pings /ping every 5 minutes. Each cycle completes in ~45 seconds.</p></div></div>
    <div class="step"><div class="step-num">7</div><div><h4>Logging &amp; evaluation</h4><p>Every cycle logged to CSV: timestamp, direction, LGBM proba, meta proba, Polymarket snapshot, BTC close. Results compared against Polymarket outcomes for ground-truth win rate evaluation.</p></div></div>
  </div>

  <div class="card">
    <div class="card-title">Tech stack</div>
    <div>
      <span class="tech-tag">Python 3.11</span><span class="tech-tag">LightGBM</span><span class="tech-tag">scikit-learn</span><span class="tech-tag">pandas</span><span class="tech-tag">numpy</span><span class="tech-tag">scipy</span><span class="tech-tag">ccxt</span><span class="tech-tag">Flask</span><span class="tech-tag">Gunicorn</span><span class="tech-tag">APScheduler</span><span class="tech-tag">Render</span><span class="tech-tag">Polymarket CLOB API</span><span class="tech-tag">Lopez de Prado AFML</span>
    </div>
  </div>
</div>

<!-- ══════════════════════════ TAB 2 — PERFORMANCE -->
<div id="tab-perf" class="tab-content">

  <div class="grid-4">
    <div class="metric"><div class="metric-label">LGBM AUC OOS</div><div class="metric-value up">0.768</div><div class="metric-sub">5-fold purged CV</div></div>
    <div class="metric"><div class="metric-label">Directional ACC</div><div class="metric-value">68.5%</div><div class="metric-sub">vs 50% random</div></div>
    <div class="metric"><div class="metric-label">Meta precision</div><div class="metric-value up">89.3%</div><div class="metric-sub">when RF confirms</div></div>
    <div class="metric"><div class="metric-label">Signal filter</div><div class="metric-value">36%</div><div class="metric-sub">of signals kept</div></div>
  </div>

  <div class="grid-4">
    <div class="metric"><div class="metric-label">Total cycles</div><div class="metric-value" id="m-total">—</div><div class="metric-sub">since deployment</div></div>
    <div class="metric"><div class="metric-label">Trades placed</div><div class="metric-value" id="m-trades">—</div><div class="metric-sub" id="m-traderate">—</div></div>
    <div class="metric"><div class="metric-label">UP signals</div><div class="metric-value up" id="m-up">—</div><div class="metric-sub">confirmed by RF</div></div>
    <div class="metric"><div class="metric-label">DOWN signals</div><div class="metric-value down" id="m-down">—</div><div class="metric-sub">confirmed by RF</div></div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-title">Simulated equity curve — $10 flat bet</div>
      <div class="chart-wrap" style="height:190px"><canvas id="eqChart" role="img" aria-label="Simulated equity curve">Equity curve.</canvas></div>
      <div style="display:flex;gap:16px;margin-top:10px;font-size:11px;color:#475569">
        <span style="display:flex;align-items:center;gap:4px"><span style="width:10px;height:2px;display:inline-block;background:#3b82f6"></span>Equity (starts $100)</span>
        <span id="eq-summary" style="margin-left:auto;color:#64748b"></span>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Feature importance — LightGBM top 8</div>
      <div id="feat-list"></div>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-title">Recent cycles</div>
      <table>
        <thead><tr>
          <th style="width:17%">Time UTC</th>
          <th style="width:15%">Side</th>
          <th style="width:17%">LGBM</th>
          <th style="width:17%">Meta</th>
          <th style="width:34%">BTC close</th>
        </tr></thead>
        <tbody id="trade-tbody"><tr><td colspan="5" style="text-align:center;color:#334155;padding:1.5rem">Loading...</td></tr></tbody>
      </table>
    </div>
    <div class="card">
      <div class="card-title">Polymarket current window</div>
      <div id="poly-live-container"></div>
      <script type="application/ld+json">
      {
        "@context": "https://schema.org",
        "@type": "WebPage",
        "name": "Bitcoin Up or Down - June 22, 8:45AM-9:00AM ET",
        "description": "Prediction market: Up 41% · Down 60% on Polymarket.",
        "url": "https://polymarket.com/event/btc-updown-15m-1782132300",
        "publisher": {
          "@type": "Organization",
          "name": "Polymarket",
          "url": "https://polymarket.com"
        }
      }
      </script>
      <figure
        class="polymarket-embed"
        id="polymarket-btc-updown-15m-1782132300"
        aria-label="Polymarket prediction market: Bitcoin Up or Down - June 22, 8:45AM-9:00AM ET"
        itemscope
        itemtype="https://schema.org/WebPage"
        style="position:relative;display:inline-block;margin:0">
        <iframe
          title="Bitcoin Up or Down - June 22, 8:45AM-9:00AM ET — Polymarket Prediction Market"
          src="https://embed.polymarket.com/market?market=btc-updown-15m-1782132300&theme=dark&border=true&height=300"
          width="400"
          height="300"
          frameborder="0"
          allowtransparency="true">
        </iframe>
        <a href="https://polymarket.com/event/btc-updown-15m-1782132300"
          aria-label="View on Polymarket"
          target="_blank"
          rel="noopener"
          style="position:absolute;top:16px;right:20px;width:120px;height:24px;z-index:10">
        </a>
        <figcaption style="position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0">
          <strong>Bitcoin Up or Down - June 22, 8:45AM-9:00AM ET</strong><br>
          Up 41% · Down 60%<br>
          <a href="https://polymarket.com/event/btc-updown-15m-1782132300">
            View full market &amp; trade on Polymarket
          </a>
        </figcaption>
      </figure>
    </div>
  </div>

  <div class="disclaimer">
    Paper trading only — no real money involved. Predictions generated by a machine learning pipeline trained on 131k Binance 1m candles (90 days). AUC 0.768 and meta precision 89.3% measured out-of-sample via Purged K-Fold CV (Lopez de Prado). Past performance does not guarantee future results.
  </div>
</div>

<!-- ══════════════════════════ TAB 3 — PROFILE -->
<div id="tab-profile" class="tab-content">
  <div class="card">
    <div style="display:flex;align-items:center;gap:18px;margin-bottom:1.5rem">
      <div class="profile-avatar">PAB</div>
      <div>
        <p style="font-size:18px;font-weight:500;color:#f1f5f9">Paul-Arnaud Battandier</p>
        <p style="font-size:13px;color:#64748b;margin-top:4px">MSc Financial Engineering &middot; ECE Paris &middot; Class of 2027</p>
        <p style="font-size:12px;color:#475569;margin-top:3px">Seeking end-of-studies internship &middot; Jan/Feb 2027 &middot; Trading or Quantitative Research</p>
      </div>
    </div>
    <div style="margin-bottom:1.5rem">
      <a class="social-btn" href="https://www.linkedin.com/in/paul-arnaud-battandier/" target="_blank">&#xea6e; LinkedIn</a>
      <a class="social-btn" href="https://github.com/Paul-Arnaud-Battandier" target="_blank">&#xea65; GitHub</a>
    </div>
    <div class="divider"></div>
    <div class="grid-2">
      <div>
        <div class="card-title">Skills</div>
        <div style="font-size:13px;display:flex;flex-direction:column;gap:14px">
          <div><div style="display:flex;justify-content:space-between;color:#cbd5e1"><span>Machine learning (LightGBM, RF, XGBoost)</span><span style="color:#475569">90%</span></div><div class="skill-bar"><div class="skill-fill" style="width:90%"></div></div></div>
          <div><div style="display:flex;justify-content:space-between;color:#cbd5e1"><span>Python (pandas, numpy, scikit-learn)</span><span style="color:#475569">85%</span></div><div class="skill-bar"><div class="skill-fill" style="width:85%"></div></div></div>
          <div><div style="display:flex;justify-content:space-between;color:#cbd5e1"><span>Quantitative finance (LdP framework)</span><span style="color:#475569">80%</span></div><div class="skill-bar"><div class="skill-fill" style="width:80%"></div></div></div>
          <div><div style="display:flex;justify-content:space-between;color:#cbd5e1"><span>API integration &amp; cloud deployment</span><span style="color:#475569">75%</span></div><div class="skill-bar"><div class="skill-fill" style="width:75%"></div></div></div>
          <div><div style="display:flex;justify-content:space-between;color:#cbd5e1"><span>Financial markets (crypto, derivatives)</span><span style="color:#475569">70%</span></div><div class="skill-bar"><div class="skill-fill" style="width:70%"></div></div></div>
        </div>
      </div>
      <div>
        <div class="card-title">This project</div>
        <p style="font-size:13px;color:#64748b;line-height:1.7;margin-bottom:1rem">Built entirely from scratch: data collection, feature engineering (Lopez de Prado), model training, live deployment on Render, and this dashboard. No boilerplate — every line written and understood.</p>
        <p style="font-size:13px;color:#64748b;line-height:1.7;margin-bottom:1rem">Started with a 4H BTC swing strategy, abandoned after backtesting revealed no edge. Pivoted to Polymarket binary prediction for a cleaner signal and verifiable ground truth.</p>
        <p style="font-size:13px;color:#64748b;line-height:1.7">Bot running live since June 2026. Every prediction logged and compared against real Polymarket outcomes.</p>
        <div style="margin-top:1rem">
          <div class="card-title" style="margin-bottom:.5rem">Looking for</div>
          <span class="tech-tag">Proprietary trading</span>
          <span class="tech-tag">Quant research</span>
          <span class="tech-tag">ML for finance</span>
          <span class="tech-tag">Paris &middot; London &middot; Remote</span>
        </div>
      </div>
    </div>
  </div>
</div>

</div>

<script>
function switchTab(id, btn) {
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  btn.classList.add('active');
}

const FEATS = [
  {name:'tf15_ret_15m_bar', v:1796, c:'#3b82f6'},
  {name:'tf15_ret_15m_bar2', v:1503, c:'#3b82f6'},
  {name:'tf15_high_low_15m', v:726,  c:'#10b981'},
  {name:'roc_30m',           v:682,  c:'#10b981'},
  {name:'tf5_ret_5m_bar',    v:664,  c:'#10b981'},
  {name:'frac_diff_03',      v:507,  c:'#f59e0b'},
  {name:'vol_30m',           v:527,  c:'#f59e0b'},
  {name:'ofi_15m',           v:295,  c:'#f59e0b'},
];
const maxV = FEATS[0].v;
const fl = document.getElementById('feat-list');
FEATS.forEach(f => {
  const pct = Math.round(f.v / maxV * 100);
  fl.innerHTML += `<div class="feat-row"><span style="color:#64748b;min-width:145px;font-size:11px;overflow:hidden;text-overflow:ellipsis">${f.name}</span><div class="feat-bar-bg"><div class="feat-bar-fill" style="width:${pct}%;background:${f.c}"></div></div><span style="font-size:11px;color:#475569;min-width:32px;text-align:right">${f.v}</span></div>`;
});

let eqChart = null, distChart = null;

async function loadData() {
  try {
    const res = await fetch('/api/trades');
    const data = await res.json();
    const trades = data.trades || [];
    const stats  = data.stats  || {};
    const bot    = data.bot_state || {};

    document.getElementById('last-update').textContent =
      'Last update: ' + (bot.last_run ? new Date(bot.last_run).toLocaleTimeString('en-GB') : '—') +
      ' · Cycles: ' + (bot.n_runs || '—');

    document.getElementById('m-total').textContent  = stats.n_total  || '0';
    document.getElementById('m-trades').textContent = stats.n_trades || '0';
    document.getElementById('m-traderate').textContent = (stats.trade_rate || '0') + '% of cycles';
    document.getElementById('m-up').textContent    = stats.up_count   || '0';
    document.getElementById('m-down').textContent  = stats.down_count || '0';

    const confirmed = trades.filter(t => t.trade === 'True');
    const last = trades[0];

    if (last) {
      document.getElementById('poly-yes').textContent = last.poly_yes_mid ? (parseFloat(last.poly_yes_mid)*100).toFixed(1)+'%' : '—';
      document.getElementById('poly-no').textContent  = last.poly_no_mid  ? (parseFloat(last.poly_no_mid)*100).toFixed(1)+'%'  : '—';
      document.getElementById('poly-spread').textContent = last.poly_spread ? parseFloat(last.poly_spread).toFixed(3) : '—';
      document.getElementById('poly-btc').textContent = last.btc_close_entry ? '$'+parseFloat(last.btc_close_entry).toLocaleString('en-US',{maximumFractionDigits:0}) : '—';
      const dec = document.getElementById('poly-decision');
      if (last.trade === 'True') {
        dec.textContent = last.direction === 'UP' ? 'BTC UP' : 'BTC DOWN';
        dec.className = last.direction === 'UP' ? 'up' : 'down';
      } else {
        dec.textContent = 'SKIP'; dec.className = 'neutral';
      }
    }

    const tbody = document.getElementById('trade-tbody');
    tbody.innerHTML = '';
    (trades.slice(0,10)).forEach(t => {
      const pc = t.side==='YES'?'pill-up':t.side==='NO'?'pill-down':'pill-skip';
      const dt = new Date(t.timestamp);
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${dt.toLocaleTimeString('en-GB',{hour:'2-digit',minute:'2-digit'})}</td><td><span class="pill ${pc}">${t.side||'SKIP'}</span></td><td>${t.lgbm_proba?(parseFloat(t.lgbm_proba)*100).toFixed(1)+'%':'—'}</td><td>${t.meta_proba&&t.meta_proba!=='None'?(parseFloat(t.meta_proba)*100).toFixed(1)+'%':'—'}</td><td>$${t.btc_close_entry?parseFloat(t.btc_close_entry).toLocaleString('en-US',{maximumFractionDigits:0}):'—'}</td>`;
      tbody.appendChild(tr);
    });

    let eq = 100;
    const eqLabels = ['Start'], eqData = [100], eqColors = [];
    confirmed.slice().reverse().forEach(t => {
      const pUp  = t.lgbm_proba ? parseFloat(t.lgbm_proba) : 0.5;
      const won  = (t.direction === 'UP' && pUp >= 0.5) || (t.direction === 'DOWN' && pUp < 0.5);
      eq += won ? 9 : -10;
      const dt = new Date(t.timestamp);
      eqLabels.push(dt.toLocaleTimeString('en-GB',{hour:'2-digit',minute:'2-digit'}));
      eqData.push(parseFloat(eq.toFixed(2)));
      eqColors.push(won ? '#10b981' : '#f87171');
    });
    document.getElementById('eq-summary').textContent = `P&L: ${eq>=100?'+':''}$${(eq-100).toFixed(0)}`;

    if (eqChart) eqChart.destroy();
    eqChart = new Chart(document.getElementById('eqChart'), {
      type: 'line',
      data: { labels: eqLabels, datasets: [{
        data: eqData, borderColor: '#3b82f6', borderWidth: 2,
        pointRadius: eqData.map((_,i) => i===0?0:4),
        pointBackgroundColor: ['#3b82f6', ...eqColors],
        fill: false, tension: 0.3
      }]},
      options: { responsive:true, maintainAspectRatio:false, plugins:{legend:{display:false}},
        scales: {
          x:{ticks:{font:{size:10},color:'#475569',maxTicksLimit:6},grid:{color:'rgba(255,255,255,0.03)'}},
          y:{ticks:{font:{size:10},color:'#475569',callback:v=>'$'+v},grid:{color:'rgba(255,255,255,0.05)'}}
        }
      }
    });

    const upC = stats.up_count||0, dnC = stats.down_count||0, skC = stats.n_skips||0;
    if (distChart) distChart.destroy();
    distChart = new Chart(document.getElementById('distChart'), {
      type: 'doughnut',
      data: { labels:['UP','DOWN','Skip'], datasets:[{data:[upC,dnC,skC],backgroundColor:['#10b981','#f87171','#334155'],borderWidth:0}]},
      options: { responsive:true, maintainAspectRatio:false, cutout:'70%', plugins:{legend:{display:false}}}
    });
    document.getElementById('dist-legend').innerHTML =
      `<span style="display:flex;align-items:center;gap:5px"><span style="width:8px;height:8px;border-radius:2px;background:#10b981"></span>UP ${upC}</span>` +
      `<span style="display:flex;align-items:center;gap:5px"><span style="width:8px;height:8px;border-radius:2px;background:#f87171"></span>DOWN ${dnC}</span>` +
      `<span style="display:flex;align-items:center;gap:5px"><span style="width:8px;height:8px;border-radius:2px;background:#334155"></span>Skip ${skC}</span>`;

  } catch(e) {
    console.error('Error loading data:', e);
  }
}

# MODIFS JS
async function loadPolymarketEmbed() {
  const container = document.getElementById("poly-live-container");
  if (!container) return;

  try {
    const res = await fetch("/api/polymarket-current");
    const data = await res.json();
    const slug = data.slug;

    container.innerHTML = `
      <figure
        class="polymarket-embed"
        id="polymarket-${slug}"
        aria-label="Polymarket prediction market"
        itemscope
        itemtype="https://schema.org/WebPage"
        style="position:relative;display:inline-block;margin:0">
        <iframe
          title="Polymarket Prediction Market"
          src="https://embed.polymarket.com/market?market=${slug}&theme=dark&border=true&height=300&t=${Date.now()}"
          width="400"
          height="300"
          frameborder="0"
          allowtransparency="true">
        </iframe>
        <a href="https://polymarket.com/event/${slug}"
          aria-label="View on Polymarket"
          target="_blank"
          rel="noopener"
          style="position:absolute;top:16px;right:20px;width:120px;height:24px;z-index:10">
        </a>
      </figure>
    `;
  } catch (e) {
    container.innerHTML = `<div style="color:#64748b">Polymarket unavailable</div>`;
  }
}



loadPolymarketEmbed();
setInterval(loadPolymarketEmbed, 30000);

loadData();
setInterval(loadData, 60000);
</script>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# DÉMARRAGE DU BOT (compatible Gunicorn + python direct)
# ---------------------------------------------------------------------------

_scheduler = start_scheduler()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Flask démarré sur port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)