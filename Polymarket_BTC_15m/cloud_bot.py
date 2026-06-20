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

# --- Ajout du répertoire racine au path ---
# __file__ = /opt/render/project/src/Polymarket_BTC_15m/cloud_bot.py
# ROOT     = /opt/render/project/src/Polymarket_BTC_15m/
# sys.path doit contenir ROOT pour que `scriptsv2.data` soit trouvable
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Sécurité : forcer aussi le dossier parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from flask import Flask, jsonify

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
# FLASK APP
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/")
def home():
    """Keep-alive endpoint — pingé par UptimeRobot toutes les 5 min."""
    with _state_lock:
        n_runs   = _bot_state["n_runs"]
        last_run = _bot_state["last_run"]
    return (
        f"🟢 BTC Polymarket Bot v2 | "
        f"Cycles: {n_runs} | "
        f"Dernier: {last_run or 'jamais'}"
    )


@app.route("/health")
def health():
    """Statut détaillé en JSON."""
    with _state_lock:
        state = dict(_bot_state)
    return jsonify({
        "status" : "running",
        "uptime" : state["started_at"],
        "stats"  : {
            "n_runs"  : state["n_runs"],
            "n_trades": state["n_trades"],
            "n_skips" : state["n_skips"],
            "last_run": state["last_run"],
        },
        "last_decision": state["last_decision"],
        "recent_errors": state["errors"][-3:],
    })


@app.route("/status")
def status():
    """Dernière décision du bot."""
    with _state_lock:
        decision = _bot_state.get("last_decision")
        last_run = _bot_state.get("last_run")
    return jsonify({
        "last_run"     : last_run,
        "last_decision": decision,
    })


@app.route("/csv")
def show_csv():
    """Affiche le trade history en HTML lisible."""
    if not TRADE_LOG.exists():
        return "⏳ Aucun trade loggé pour l'instant."

    with open(TRADE_LOG, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        return "⏳ Fichier CSV vide."

    # Construction table HTML
    header = lines[0].strip().split(",")
    rows   = [l.strip().split(",") for l in lines[1:]]

    # Dernières lignes en premier
    rows.reverse()

    th = "".join(f"<th>{h}</th>" for h in header)
    trs = ""
    for row in rows:
        side  = row[header.index("side")]  if "side"  in header else ""
        trade = row[header.index("trade")] if "trade" in header else ""
        color = "#d4edda" if trade == "True" else "#f8f9fa"
        td    = "".join(f"<td>{v}</td>" for v in row)
        trs  += f'<tr style="background:{color}">{td}</tr>'

    n_trades = sum(1 for r in rows if len(r) > header.index("trade") and r[header.index("trade")] == "True")
    n_total  = len(rows)
    rate     = f"{n_trades/n_total*100:.1f}%" if n_total > 0 else "N/A"

    html = f"""
    <!DOCTYPE html><html><head>
    <meta charset="utf-8">
    <title>BTC Bot — Trade History</title>
    <meta http-equiv="refresh" content="60">
    <style>
      body {{ font-family: monospace; padding: 20px; background: #1a1a2e; color: #eee; }}
      h1 {{ color: #00d4ff; }}
      .stats {{ background: #16213e; padding: 12px; border-radius: 8px; margin-bottom: 16px; }}
      table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
      th {{ background: #0f3460; color: #00d4ff; padding: 8px; text-align: left; }}
      td {{ padding: 6px 8px; border-bottom: 1px solid #333; }}
      .badge {{ padding: 2px 8px; border-radius: 4px; font-weight: bold; }}
    </style>
    </head><body>
    <h1>🤖 BTC Polymarket Bot — Trade History</h1>
    <div class="stats">
      📊 Total cycles : <b>{n_total}</b> |
      ✅ Trades : <b>{n_trades}</b> |
      📈 Trade rate : <b>{rate}</b> |
      🔄 Refresh auto : 60s
    </div>
    <table>
      <thead><tr>{th}</tr></thead>
      <tbody>{trs}</tbody>
    </table>
    </body></html>
    """
    return html


@app.route("/run_now")
def run_now():
    """Déclenche un cycle manuellement (debug)."""
    thread = threading.Thread(target=run_cycle)
    thread.daemon = True
    thread.start()
    return jsonify({"status": "cycle lancé en arrière-plan"})


# ---------------------------------------------------------------------------
# DÉMARRAGE DU BOT (compatible Gunicorn + python direct)
# ---------------------------------------------------------------------------

# Lance le scheduler dans un thread daemon dès l'import du module
# → Gunicorn importe cloud_bot, le bot démarre automatiquement
_scheduler = start_scheduler()

# Premier cycle immédiat au démarrage (pour tester que tout fonctionne)
_init_thread = threading.Thread(target=run_cycle)
_init_thread.daemon = True
_init_thread.start()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Flask démarré sur port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)