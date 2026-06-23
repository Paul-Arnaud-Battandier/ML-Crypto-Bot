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
from flask import send_file


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

from Polymarket_BTC_15m.scriptsv2.data.polymarket_feed import get_current_15m_slug

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
        from Polymarket_BTC_15m.scriptsv2.data.binance_feed import fetch_live
        df_live = fetch_live(n_bars=90)

        if df_live is None or len(df_live) < 35:
            logger.error(f"Données Binance insuffisantes : {len(df_live) if df_live is not None else 0} bougies")
            return

        btc_close = float(df_live["close"].iloc[-1])
        logger.info(f"BTC close : {btc_close:,.2f} USDT | {len(df_live)} bougies 1m")

        # ------------------------------------------------------------------
        # 2. Features
        # ------------------------------------------------------------------
        from Polymarket_BTC_15m.scriptsv2.features.feature_pipeline import build_live_features
        live_features = build_live_features(df_live)

        if live_features is None:
            logger.error("Impossible de construire les features live")
            return

        # ------------------------------------------------------------------
        # 3. Chargement modèles
        # ------------------------------------------------------------------
        from Polymarket_BTC_15m.scriptsv2.training.train_lgbm import load_model
        from Polymarket_BTC_15m.scriptsv2.training.train_meta import load_meta_model

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
            from Polymarket_BTC_15m.scriptsv2.data.polymarket_feed import get_market_snapshot
            poly_snapshot = get_market_snapshot()
            if poly_snapshot:
                logger.info(f"Polymarket | YES={poly_snapshot.get('yes_mid', 'N/A')} | "
                            f"NO={poly_snapshot.get('no_mid', 'N/A')}")
        except Exception as e:
            logger.warning(f"Polymarket indisponible : {e} — on continue sans")

        # ------------------------------------------------------------------
        # 5. Décision combinée
        # ------------------------------------------------------------------
        from Polymarket_BTC_15m.scriptsv2.training.train_meta import predict_combined

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
        from Polymarket_BTC_15m.scriptsv2.data.polymarket_feed import get_current_15m_slug
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


@app.route("/download_csv")
def download_csv():
    """Permet de télécharger l'historique complet des trades."""
    if TRADE_LOG.exists():
        return send_file(TRADE_LOG, as_attachment=True, download_name="polymarket_trades.csv")
    return "Aucune donnée pour le moment", 404


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
.tabs{display:flex;gap:0;justify-content:center;border-bottom:1px solid #1e2432;margin-bottom:2rem}
.tab{padding:10px 20px;font-size:13px;font-weight:500;color:#64748b;cursor:pointer;border:none;background:none;border-bottom:2px solid transparent;margin-bottom:-1px}
.tab:hover{color:#94a3b8}
.tab.active{color:#f1f5f9;border-bottom:2px solid #fb8b1e}
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
/* RESPONSIVE DESIGN (Mobiles & Tablettes) */
@media (max-width: 768px) {
  .header { flex-direction: column; align-items: flex-start; gap: 15px; position: relative; }
  .badge-live { position: absolute; top: 0; right: 0; }
  .grid-4 { grid-template-columns: repeat(2, 1fr); gap: 10px; }
  .grid-2 { grid-template-columns: 1fr; gap: 15px; }
  .tabs { justify-content: space-between; }
  .tab { padding: 10px 12px; font-size: 12px; flex: 1; text-align: center; }
  
  /* Permet au tableau de slider de gauche à droite sur mobile sans casser la page */
  table { display: block; overflow-x: auto; white-space: nowrap; }
  
  /* Ajuste l'en-tête de l'onglet Profile pour éviter le chevauchement */
  #tab-profile > .card > div:first-child { flex-direction: column; align-items: flex-start; text-align: left; }
}
</style>
</head>
<body>
<div class="container">

<div class="header">
  <div class="header-left">
    <h1><span style="color:#fb8b1e">Algorithmic Trading Bot</span> — BTC 15 mins candles on Polymarket</h1>
    <p>LightGBM + Random Forest meta-labelling &middot; Paper trading</p>
  </div>
  <span class="badge-live">Live</span>
</div>

<div class="tabs">
  <button class="tab" onclick="switchTab('approach',this)">Approach</button>
  <button class="tab active" onclick="switchTab('perf',this)">Performance</button>
  <button class="tab" onclick="switchTab('profile',this)">About me</button>
</div>

<!-- ══════════════════════════ TAB 1 — APPROACH -->
<div id="tab-approach" class="tab-content">
  
  <div class="card" style="border-left: 3px solid #f87171;">
    <div class="card-title" style="color:#f1f5f9;">Phase 1 — BTC/USDT 4H Swing Trading <span style="color:#f87171;">(Abandoned)</span></div>
    
    <div style="display:flex; flex-direction:column; gap:12px;">
      <div style="font-size:13px; color:#cbd5e1; line-height:1.6;">
        <div style="color:#fb8b1e; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Research & Strategy</div>
        <strong>Asset:</strong> BTC/USDT. <strong>Market Characteristics:</strong> 24/7/365 trading environment, heavily retail-driven, distinct cyclical regimes. <strong>Horizon:</strong> Swing Trading. <strong>Drivers:</strong> Funding Rates, on-chain flows, macroeconomic cyclicality. <strong>Capital:</strong> 100€ initial budget (Forward-testing via Binance Testnet). <strong>Costs:</strong> Binance Futures (Maker: 0.02% | Taker: 0.05%). <strong>Risks:</strong> Look-ahead bias, Overfitting (Curve-fitting).
      </div>

      <div style="font-size:13px; color:#cbd5e1; line-height:1.6;">
        <div style="color:#fb8b1e; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Data</div>
        <strong>Market:</strong> OHLCV. <strong>Alternative:</strong> Funding Rate, Fear & Greed Index. <strong>Macro:</strong> DXY, Equities (via yfinance). <strong>Excluded:</strong> Open Interest (historical data unavailable at required granularity).
      </div>

      <div style="font-size:13px; color:#cbd5e1; line-height:1.6;">
        <div style="color:#fb8b1e; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Features</div>
        <strong>Raw Inputs:</strong> 9 base features. <strong>Engineered:</strong> 15 quantitative features (Stationary transformations, volatility metrics, relative distances).
      </div>

      <div style="font-size:13px; color:#cbd5e1; line-height:1.6;">
        <div style="color:#fb8b1e; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Training</div>
        <em>Phase 1 (Directional):</em> XGBoost (outperformed LightGBM), 4H horizon predicting T+24H (rolling window), Optuna TPE tuning. <br>
        <em>Phase 2 (Meta-Labeling):</em> Random Forest Classifier (n_estimators=200, class_weight='balanced'). Triple Barrier Setup: PT = 1.5x Vol, SL = 1.0x Vol, TL = 24h. <br>
        Break-even required > 40.0% precision. Primary model alone yielded 33.25% (Unprofitable). Meta-Model reached 55.14% precision at 54.0% confidence threshold.
      </div>

      <div style="font-size:13px; color:#cbd5e1; line-height:1.6; background:#2d0a0a; border:1px solid #7f1d1d; padding:12px; border-radius:6px; margin-top:8px;">
        <div style="color:#f87171; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Backtest & The Pivot</div>
        Ultimately, the models combined did not make a positive return. The 4H market exhibited too much of a "Random Walk" with severe intrabar noise. It was impossible at my scale to construct a robust, winning strategy without massive capital to absorb drawdowns. <br>
        <strong>Decision: Pivot to 15-minute BTC Up or Down binary contracts on Polymarket.</strong>
      </div>
    </div>
  </div>

  <div class="card" style="border-left: 3px solid #10b981;">
    <div class="card-title" style="color:#f1f5f9;">Phase 2 — Polymarket 15-Minute Binary Bot <span style="color:#10b981;">(Live)</span></div>
    
    <div style="display:flex; flex-direction:column; gap:12px;">
      <div style="font-size:13px; color:#cbd5e1; line-height:1.6;">
        <div style="color:#fb8b1e; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Research & Strategy</div>
        <strong>Asset:</strong> Polymarket BTC Up/Down Contracts. <strong>Horizon:</strong> 15 minutes. <strong>Advantage:</strong> Eliminates stop-loss hunting, slippage, and complex position sizing. The binary label resolves precisely at close[t+15] > close[t], providing a mathematically pristine environment for the algorithm.
      </div>

      <div style="font-size:13px; color:#cbd5e1; line-height:1.6;">
        <div style="color:#fb8b1e; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Data</div>
        <strong>Memory:</strong> Binance OHLCV 1m API (Incremental updates). <strong>Execution:</strong> Polymarket Gamma API (Dynamic token IDs) & CLOB API (Order book matching).
      </div>

      <div style="font-size:13px; color:#cbd5e1; line-height:1.6;">
        <div style="color:#fb8b1e; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Features</div>
        Engineered specific short-term inertia indicators: Rolling Volatility (60m std), Momentum (5m & 15m returns), Volume Surge (vs 4H mean), and RSI (14-period). Absolute prices were strictly excluded to prevent look-ahead bias and memorization.
      </div>

      <div style="font-size:13px; color:#cbd5e1; line-height:1.6;">
        <div style="color:#fb8b1e; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Training</div>
        Trained a LightGBM Classifier on parsed historical data. The model learns the probability of a bullish outcome based purely on the structured mathematical features.
      </div>

      <div style="font-size:13px; color:#cbd5e1; line-height:1.6;">
        <div style="color:#fb8b1e; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Backtest</div>
        Achieved a baseline directional accuracy of ~50.36%. In a binary market with dynamic odds, an edge of +0.36% over a coin flip translates to profitability when identifying and buying mispriced contracts (e.g., buying a 50.36% true-probability "YES" contract when the market prices it at 40¢).
      </div>

      <div style="font-size:13px; color:#cbd5e1; line-height:1.6;">
        <div style="color:#fb8b1e; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Deployment</div>
        Autonomous Python architecture deployed on Render. Scheduled via APScheduler to execute precisely at hh:02, hh:17, hh:32, hh:47, ensuring Binance candles are fully closed and new Polymarket contracts have absorbed initial liquidity.
      </div>

      <div style="font-size:13px; color:#cbd5e1; line-height:1.6;">
        <div style="color:#fb8b1e; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Oversight</div>
        Flask web dashboard acting as a live terminal. Real-time logging of prediction confidence and dynamic risk shielding (aborts trade instantly if the Polymarket bid/ask spread exceeds $0.05).
      </div>
    </div>
  </div>

  <div class="card">
    <div class="card-title" style="color:#f1f5f9;">Sources & Acknowledgements</div>
    <div style="display:flex; gap:10px; flex-wrap:wrap; margin-top:10px;">
      <span class="tech-tag" style="border-color:#fb8b1e; color:#fb8b1e;">Advances in Financial Machine Learning (Marcos Lopez de Prado)</span>
      <span class="tech-tag">Gemini Pro</span>
      <span class="tech-tag">Claude</span>
    </div>
  </div>

</div>

<!-- ══════════════════════════ TAB 2 — PERFORMANCE -->
<div id="tab-perf" class="tab-content active">

  <div class="grid-4">
    <div class="metric"><div class="metric-label">Total cycles</div><div class="metric-value" id="m-total">—</div><div class="metric-sub" id="m-since">since deployment</div></div>
    <div class="metric"><div class="metric-label">Trades placed</div><div class="metric-value" id="m-trades">—</div><div class="metric-sub" id="m-traderate">—</div></div>
    <div class="metric"><div class="metric-label">UP signals</div><div class="metric-value up" id="m-up">—</div><div class="metric-sub">confirmed by RF</div></div>
    <div class="metric"><div class="metric-label">DOWN signals</div><div class="metric-value down" id="m-down">—</div><div class="metric-sub">confirmed by RF</div></div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-title" style="display:flex;justify-content:space-between;align-items:center;">
        Recent cycles
        <a href="/download_csv" class="social-btn" style="padding:4px 10px;font-size:11px;margin:0;color:#fb8b1e;border-color:#fb8b1e;">&#x2193; Download CSV</a>
      </div>
      <table>
        <thead><tr>
          <th style="width:20%">Time UTC</th>
          <th style="width:15%">LGBM</th>
          <th style="width:15%">Meta</th>
          <th style="width:20%">Side</th>
          <th style="width:30%">Result</th>
        </tr></thead>
        <tbody id="trade-tbody"><tr><td colspan="5" style="text-align:center;color:#334155;padding:1.5rem">Loading...</td></tr></tbody>
      </table>
    </div>
    <div class="card">
      <div class="card-title">Polymarket current window</div>
      <div id="poly-live-container"></div>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-title">Simulated equity curve — $10 flat bet</div>
      <div class="chart-wrap" style="height:190px"><canvas id="eqChart"></canvas></div>
      <div style="display:flex;gap:16px;margin-top:10px;font-size:11px;color:#475569">
        <span style="display:flex;align-items:center;gap:4px"><span style="width:10px;height:2px;display:inline-block;background:#fb8b1e"></span>Equity (starts $100)</span>
        <span id="eq-summary" style="margin-left:auto;color:#64748b"></span>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Live Signal Distribution</div>
      <div class="chart-wrap" style="height:170px;margin-top:10px;"><canvas id="distChart"></canvas></div>
      <div id="dist-legend" style="display:flex;justify-content:center;gap:15px;margin-top:15px;font-size:11px;color:#cbd5e1"></div>
    </div>
  </div>

  <div class="disclaimer">
    Paper trading only — no real money involved. Past performance does not guarantee future results.
  </div>
</div>

<!-- ══════════════════════════ TAB 3 — PROFILE -->
<div id="tab-profile" class="tab-content">
  <div class="card">
    <div style="display:flex;align-items:center;gap:18px;margin-bottom:1.5rem">
      <div class="profile-avatar">PAB</div>
      <div>
        <p style="font-size:22px;font-weight:600;color:#f1f5f9">Paul Arnaud-Battandier</p>
        <p style="font-size:14px;color:#cbd5e1;margin-top:4px">Major in Finance & Quantitative Engineering &middot; ECE Paris</p>
        <p style="font-size:15px;color:#fb8b1e;margin-top:6px;font-weight:500;">Seeking an end-of-studies internship &middot; Jan/Feb 2027 &middot; Trading or Quantitative Research</p>
      </div>
    </div>

    <div style="margin-bottom:1.5rem; display:flex; gap:12px; flex-wrap:wrap;">
      <a class="social-btn" href="https://www.linkedin.com/in/paul-arnaud-battandier/" target="_blank">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path><rect x="2" y="9" width="4" height="12"></rect><circle cx="4" cy="4" r="2"></circle></svg>
        LinkedIn
      </a>
      <a class="social-btn" href="https://github.com/Paul-Arnaud-Battandier" target="_blank">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path></svg>
        GitHub
      </a>
      <a class="social-btn" href="/static/CV_Paul_Arnaud-Battandier.pdf" target="_blank" style="border-color:#fb8b1e; color:#fb8b1e; background:rgba(251, 139, 30, 0.05);">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
        Download CV
      </a>
    </div>

    <div class="divider"></div>

    <div style="display:flex; flex-direction:column; gap:2rem; margin-top:1.5rem;">
      
      <div>
        <div class="card-title" style="color:#fb8b1e; font-size:12px;">About my personality</div>
        <p style="font-size:14px;color:#cbd5e1;line-height:1.7;">I am highly analytical, pragmatic, and driven by complex problem-solving. I thrive at the intersection of mathematics, computer science, and financial markets. I believe in building robust, data-driven systems without "black box" illusions, always focusing on statistical edge, strict risk management, and clean architecture.</p>
      </div>

      <div>
        <div class="card-title" style="color:#fb8b1e; font-size:12px;">My core skills</div>
        <p style="font-size:14px;color:#cbd5e1;line-height:1.7;">My technical stack is centered around Quantitative Finance and Machine Learning. I specialize in Python (Pandas, NumPy, Scikit-Learn) and predictive modeling (LightGBM, Random Forest, XGBoost). I am heavily influenced by Marcos Lopez de Prado's framework for <em>Advances in Financial Machine Learning</em> (Purged CV, fractional differentiation, meta-labeling). Beyond research, I build end-to-end pipelines: from API integrations to live cloud deployments.</p>
      </div>

      <div>
        <div class="card-title" style="color:#fb8b1e; font-size:12px;">My ambitions</div>
        <p style="font-size:14px;color:#cbd5e1;line-height:1.7;">My immediate goal is to secure an end-of-studies internship in a fast-paced proprietary trading firm, hedge fund, or quantitative research desk. I want to surround myself with industry experts, contribute to alpha generation or execution optimization, and ultimately evolve into a top-tier Quantitative Researcher or Algorithmic Trader.</p>
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

let eqChart = null, distChart = null;

async function loadData() {
  try {
    const res = await fetch('/api/trades');
    const data = await res.json();
    const trades = data.trades || [];
    const stats  = data.stats  || {};

    // 1. Mise à jour des compteurs principaux
    document.getElementById('m-total').textContent  = stats.n_total  || '0';
    document.getElementById('m-trades').textContent = stats.n_trades || '0';
    document.getElementById('m-up').textContent    = stats.up_count   || '0';
    document.getElementById('m-down').textContent  = stats.down_count || '0';

    // 2. Mise à jour dynamique du texte "since [Date, Heure]"
    if (trades.length > 0) {
      const firstTradeDt = new Date(trades[trades.length - 1].timestamp);
      // Format : "Jun 22, 16:47 UTC"
      const dStr = firstTradeDt.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' });
      const tStr = firstTradeDt.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
      const timeString = `${dStr}, ${tStr} UTC`;
      
      document.getElementById('m-since').textContent = 'since ' + timeString;
      document.getElementById('m-traderate').textContent = 'since ' + timeString;
    } else {
      document.getElementById('m-since').textContent = 'awaiting data...';
      document.getElementById('m-traderate').textContent = 'awaiting data...';
    }

    const confirmed = trades.filter(t => t.trade === 'True');

    // 3. Mise à jour du Tableau Recent Cycles (Ordre modifié + Pilule Result)
    const tbody = document.getElementById('trade-tbody');
    tbody.innerHTML = '';
    if (trades.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:#64748b;padding:1.5rem">No cycles recorded yet.</td></tr>';
    } else {
      const now = new Date();
      (trades.slice(0,10)).forEach(t => {
        const dt = new Date(t.timestamp);
        
        // Formatage des données
        const lgbmStr = t.lgbm_proba ? (parseFloat(t.lgbm_proba)*100).toFixed(1)+'%' : '—';
        const metaStr = t.meta_proba && t.meta_proba !== 'None' ? (parseFloat(t.meta_proba)*100).toFixed(1)+'%' : '—';
        const pc = t.side==='YES' ? 'pill-up' : t.side==='NO' ? 'pill-down' : 'pill-skip';
        
        // Logique de la pilule Result
        let resultHtml = '<span style="color:#64748b">—</span>';
        if (t.trade === 'True') {
          const isFinished = (now.getTime() - dt.getTime()) >= 15 * 60 * 1000; // 15 mins écoulées ?
          
          if (!isFinished) {
            resultHtml = `<span class="pill" style="background:#1e293b;color:#94a3b8;border:1px solid #334155">Pending</span>`;
          } else {
            // Logique simulée (Mock) en attendant un vrai backend
            const pUp = t.lgbm_proba ? parseFloat(t.lgbm_proba) : 0.5;
            const won = (t.direction === 'UP' && pUp >= 0.5) || (t.direction === 'DOWN' && pUp < 0.5);
            if (won) {
              resultHtml = `<span class="pill pill-up">WIN</span>`;
            } else {
              resultHtml = `<span class="pill pill-down">LOSS</span>`;
            }
          }
        }

        const tr = document.createElement('tr');
        // Nouvel ordre : Time | LGBM | Meta | Side | Result
        tr.innerHTML = `
          <td>${dt.toLocaleTimeString('en-GB',{hour:'2-digit',minute:'2-digit'})}</td>
          <td>${lgbmStr}</td>
          <td>${metaStr}</td>
          <td><span class="pill ${pc}">${t.side||'SKIP'}</span></td>
          <td>${resultHtml}</td>
        `;
        tbody.appendChild(tr);
      });
    }

    // 4. Graphique : Equity Curve
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
        data: eqData, borderColor: '#fb8b1e', borderWidth: 2,
        pointRadius: eqData.map((_,i) => i===0?0:4),
        pointBackgroundColor: ['#fb8b1e', ...eqColors],
        fill: false, tension: 0.3
      }]},
      options: { responsive:true, maintainAspectRatio:false, plugins:{legend:{display:false}},
        scales: {
          x:{ticks:{font:{size:10},color:'#475569',maxTicksLimit:6},grid:{color:'rgba(255,255,255,0.03)'}},
          y:{ticks:{font:{size:10},color:'#475569',callback:v=>'$'+v},grid:{color:'rgba(255,255,255,0.05)'}}
        }
      }
    });

    // 5. Graphique : Signal Distribution
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

// 6. Widget Polymarket
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
        style="position:relative;display:inline-block;margin:0;width:100%;">
        <iframe
          src="https://embed.polymarket.com/market?market=${slug}&theme=dark&border=true&height=300"
          style="width:100%;height:300px;border:none;"
          frameborder="0"
          allowtransparency="true">
        </iframe>
      </figure>
    `;
  } catch (e) {
    container.innerHTML = `<div style="color:#64748b;text-align:center;padding:2rem;">Awaiting next Polymarket window...</div>`;
  }
}

// Lancement automatique
loadPolymarketEmbed();
setInterval(loadPolymarketEmbed, 15000);

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