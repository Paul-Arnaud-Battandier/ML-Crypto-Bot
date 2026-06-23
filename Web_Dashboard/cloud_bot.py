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
    """Dashboard principal — Focus sur la narration du portfolio Quant."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Algorithmic Trading Bot</title>
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
.badge-wip{display:inline-flex;align-items:center;gap:5px;background:#1e3a5f;color:#60a5fa;font-size:11px;padding:4px 10px;border-radius:20px;border:1px solid #2563eb}
.tabs{display:flex;gap:0;justify-content:center;border-bottom:1px solid #1e2432;margin-bottom:2rem}
.tab{padding:10px 20px;font-size:13px;font-weight:500;color:#64748b;cursor:pointer;border:none;background:none;border-bottom:2px solid transparent;margin-bottom:-1px}
.tab:hover{color:#94a3b8}
.tab.active{color:#f1f5f9;border-bottom:2px solid #fb8b1e}
.tab-content{display:none}.tab-content.active{display:block}
.card{background:#141820;border:1px solid #1e2432;border-radius:12px;padding:1.25rem;margin-bottom:1.25rem}
.card-title{font-size:11px;font-weight:500;color:#475569;text-transform:uppercase;letter-spacing:.06em;margin-bottom:1rem}
.tech-tag{display:inline-block;font-size:11px;padding:3px 9px;border-radius:5px;background:#1e2432;color:#64748b;margin:2px;border:1px solid #2d3748}
/* RESPONSIVE DESIGN */
@media (max-width: 768px) {
  .header { flex-direction: column; align-items: flex-start; gap: 15px; }
  .badge-wip { align-self: flex-start; }
  .tabs { justify-content: space-between; width: 100%; }
  .tab { padding: 10px 12px; font-size: 12px; flex: 1; text-align: center; }
}
</style>
</head>
<body>
<div class="container">

<div class="header">
  <div class="header-left">
    <h1><span style="color:#fb8b1e">Algorithmic Trading Bot</span> — </h1>
    <p>Quantitative Research Portfolio &middot; Python, Machine Learning & Statistics</p>
  </div>
  <span class="badge-wip">Research Phase</span>
</div>

<div class="tabs">
  <button class="tab active" onclick="switchTab('approach',this)">Approach & Journey</button>
  <button class="tab" onclick="switchTab('perf',this)">Performance (WIP)</button>
  <button class="tab" onclick="switchTab('profile',this)">About me</button>
</div>

<div id="tab-approach" class="tab-content active">
  
  <div class="card" style="border-left: 3px solid #f87171;">
    <div class="card-title" style="color:#f1f5f9;">Phase 1 — BTC/USDT 4H Swing Trading <span style="color:#f87171;">(Abandoned)</span></div>
    
    <div style="display:flex; flex-direction:column; gap:12px;">
      <div style="font-size:13px; color:#cbd5e1; line-height:1.6;">
        <div style="color:#fb8b1e; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Research & Strategy</div>
        <strong>Asset:</strong> BTC/USDT. <strong>Horizon:</strong> Swing Trading. <strong>Drivers:</strong> Funding Rates, macro cyclicality. <strong>Risks:</strong> Look-ahead bias, Curve-fitting.
      </div>
      <div style="font-size:13px; color:#cbd5e1; line-height:1.6;">
        <div style="color:#fb8b1e; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Training & Outcome</div>
        <em>Phase 1:</em> XGBoost predicting T+24H. <em>Phase 2 (Meta-Labeling):</em> Random Forest Classifier (Triple Barrier Setup). Break-even required > 40.0% precision. Primary model yielded 33.25% (Unprofitable).
      </div>
      <div style="font-size:13px; color:#cbd5e1; line-height:1.6; background:#2d0a0a; border:1px solid #7f1d1d; padding:12px; border-radius:6px; margin-top:8px;">
        <div style="color:#f87171; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Backtest & The Pivot</div>
        The 4H market exhibited too much of a "Random Walk" with severe intrabar noise. It was impossible to construct a robust, winning strategy without massive capital to absorb stop-loss hunting.<br>
        <strong>Decision: Pivot to short-term binary options on Polymarket to eliminate slippage and stop-loss mechanics.</strong>
      </div>
    </div>
  </div>

  <div class="card" style="border-left: 3px solid #f87171;">
    <div class="card-title" style="color:#f1f5f9;">Phase 2 — Polymarket 15-Minute Binary Bot <span style="color:#f87171;">(Abandoned)</span></div>
    
    <div style="display:flex; flex-direction:column; gap:12px;">
      <div style="font-size:13px; color:#cbd5e1; line-height:1.6;">
        <div style="color:#fb8b1e; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Research & Strategy</div>
        <strong>Asset:</strong> Polymarket BTC Up/Down Contracts. <strong>Advantage:</strong> The binary label resolves precisely at close[t+15] > close[t], capping risk purely to the premium paid.
      </div>
      <div style="font-size:13px; color:#cbd5e1; line-height:1.6;">
        <div style="color:#fb8b1e; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Implementation</div>
        Trained a LightGBM Classifier on parsed historical data. Deployed an autonomous Python architecture with an APScheduler triggering predictions at strictly defined timestamps, shielded by a Random Forest meta-labeler.
      </div>
      <div style="font-size:13px; color:#cbd5e1; line-height:1.6; background:#2d0a0a; border:1px solid #7f1d1d; padding:12px; border-radius:6px; margin-top:8px;">
        <div style="color:#f87171; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Live Diagnosis & The Pivot</div>
        Market microstructure and liquidity constraints on Polymarket invalidate high-frequency directional trading. Target market (BTC 4 PM ET) showed a spread of $0.98. Risking $0.99 to gain $0.01 requires a >99.1% win rate, making the strategy mathematically unviable despite a predictive ML edge.<br>
        <strong>Decision: Pivot to Statistical Arbitrage (Pairs Trading) on Binance Futures to regain execution quality.</strong>
      </div>
    </div>
  </div>

  <div class="card" style="border-left: 3px solid #3b82f6;">
    <div class="card-title" style="color:#f1f5f9;">Phase 3 — Statistical Arbitrage (StatArb) BTC / ... <span style="color:#60a5fa;">(In Research)</span></div>
    
    <div style="display:flex; flex-direction:column; gap:12px;">
      <div style="font-size:13px; color:#cbd5e1; line-height:1.6;">
        <div style="color:#fb8b1e; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Concept & Strategy</div>
        Transitioning to a <strong>Market Neutral</strong> approach (Pairs Trading). By identifying two strongly co-integrated assets (e.g., BTC and ETH), the algorithm trades the "spread" (the mathematical divergence between them) rather than attempting to predict absolute price direction.
      </div>
      <div style="font-size:13px; color:#cbd5e1; line-height:1.6;">
        <div style="color:#fb8b1e; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">The Edge</div>
        This strategy eliminates directional market risk (immune to sudden market-wide crashes) and capitalizes on perfect execution infrastructure: infinite liquidity and $0.0001 spreads via Binance Futures.
      </div>
      <div style="font-size:13px; color:#cbd5e1; line-height:1.6;">
        <div style="color:#fb8b1e; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Current Roadmap</div>
        Conducting Engle-Granger Co-integration tests on historical OHLCV data. Next steps involve Z-Score spread modeling and building the dual-leg execution logic.
      </div>
    </div>
  </div>

</div>

<div id="tab-perf" class="tab-content">
  <div style="text-align:center; padding: 3rem 1rem; color: #64748b;">
    <h3>Engineering in progress...</h3>
    <p style="margin-top: 10px; font-size: 14px;">The infrastructure is currently being re-wired for Binance Futures WebSockets and Co-integration metrics.</p>
  </div>
</div>

<div id="tab-profile" class="tab-content">
  <div style="text-align:center; padding: 3rem 1rem; color: #64748b;">
    <p>Profil section preserved.</p>
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