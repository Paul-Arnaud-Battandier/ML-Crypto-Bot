"""
background_jobs.py
───────────────────
Lance StatArb, Funding Carry, et leurs rescans en arrière-plan, DANS le
process Flask (pour rester gratuit sur Render), mais chaque exécution
tourne en SOUS-PROCESSUS isolé plutôt qu'importée directement.

Pourquoi ce changement (vs l'ancienne version qui faisait
`from live_bot import main as bot_main` et tournait en boucle infinie
dans un thread) :
  - Un import direct charge pandas/numpy/lightgbm/ccxt en mémoire pour
    TOUJOURS dans le process principal, même entre deux scans. Cumulé
    sur 2 bots + le dashboard, ça dépassait les 512MB de Render → OOM
    kill toutes les ~20-30 minutes (voir logs du 31/08).
  - live_bot.py et live_funding_bot.py sont maintenant des scripts
    "one-shot" (un scan, une action, fin). En sous-processus, toute
    leur mémoire est rendue à l'OS dès qu'ils se terminent — le process
    principal ne retient jamais rien de lourd.
  - Render (contrairement aux runners GitHub Actions, hébergés sur IP
    US bloquées par Binance — erreur 451) n'a pas ce problème d'accès :
    les bots y fonctionnaient déjà. Le seul souci était la mémoire,
    donc on règle ÇA spécifiquement plutôt que de changer d'hébergeur.

⚠️ IMPORTANT : ne JAMAIS lancer live_bot.py / live_funding_bot.py en
local en même temps que ce service tourne sur Render — double bot sur
le même compte Binance.
"""

import sys
import time
import random
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timezone

# ── Chemins vers les scripts ────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent  # ML_Crypto_Bot/

STATARB_SCRIPT        = ROOT_DIR / "StatArb_ETH_15m"    / "scripts" / "live_bot.py"
FUNDING_SCRIPT         = ROOT_DIR / "FundingCarry_Multi" / "scripts" / "live_funding_bot.py"
SELECT_PAIR_SCRIPT     = ROOT_DIR / "Regime_Detector"    / "scripts" / "select_pair.py"
SELECT_FUNDING_SCRIPT  = ROOT_DIR / "FundingCarry_Multi" / "scripts" / "select_funding_pair.py"


def run_subprocess(script_path, label, timeout=280):
    """
    Lance un script en sous-processus isolé (StatArb, Funding, ou un
    rescan). Toute la mémoire utilisée est rendue à l'OS dès que le
    sous-processus se termine — jamais résidente dans le process
    principal Flask.
    """
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.stdout:
            print(result.stdout)
        if result.returncode != 0:
            print(f"[BG] ⚠️ {label} a terminé avec le code {result.returncode}")
            if result.stderr:
                print(result.stderr[-1500:])
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"[BG] ⚠️ {label} a dépassé le timeout ({timeout}s)")
        return False
    except Exception as e:
        print(f"[BG] ⚠️ Erreur lancement {label} : {e}")
        return False


def _seconds_until_next_quarter_hour(buffer_seconds=5):
    """Renvoie le nb de secondes jusqu'au prochain XX:00/15/30/45 (+buffer)."""
    now = datetime.now()
    minutes_to_next = 15 - (now.minute % 15)
    target = now.replace(second=0, microsecond=0)
    target = target.replace(minute=(now.minute // 15) * 15)
    from datetime import timedelta
    target = target + timedelta(minutes=15 if minutes_to_next != 15 or now.second > 0 else 0)
    wait = (target - now).total_seconds()
    if wait <= 0:
        wait += 15 * 60
    return wait + buffer_seconds


def _seconds_until_next_funding_slot(buffer_seconds=120):
    """Renvoie le nb de secondes jusqu'au prochain 00:00 / 08:00 / 16:00 UTC
    (+2min de buffer pour être sûr que le paiement de funding est passé)."""
    now_utc = datetime.now(timezone.utc)
    slots = [0, 8, 16]
    candidates = [now_utc.replace(hour=h, minute=0, second=0, microsecond=0) for h in slots]
    from datetime import timedelta
    candidates += [c + timedelta(days=1) for c in candidates]
    future = [c for c in candidates if c > now_utc]
    target = min(future)
    wait = (target - now_utc).total_seconds()
    return wait + buffer_seconds


def statarb_loop():
    """Lance live_bot.py toutes les 15 min, aligné sur les bougies."""
    time.sleep(random.uniform(10, 20))  # décale légèrement du funding_loop
    print("[BG] 🤖 StatArb loop démarré (subprocess, toutes les 15min)")
    while True:
        wait = _seconds_until_next_quarter_hour()
        time.sleep(wait)
        run_subprocess(STATARB_SCRIPT, "StatArb scan")


def funding_loop():
    """Lance live_funding_bot.py à chaque créneau de funding (00/08/16h UTC)."""
    print("[BG] 💰 Funding loop démarré (subprocess, 00h/08h/16h UTC)")
    while True:
        wait = _seconds_until_next_funding_slot()
        time.sleep(wait)
        run_subprocess(FUNDING_SCRIPT, "Funding cycle")


def statarb_rescan_loop():
    """Relance le scan de cointégration toutes les 7 jours."""
    while True:
        time.sleep(7 * 24 * 3600)
        print("[BG] 🔄 Rescan StatArb hebdomadaire...")
        run_subprocess(SELECT_PAIR_SCRIPT, "select_pair.py", timeout=600)


def funding_rescan_loop():
    """Relance le scan de sélection funding toutes les 24h."""
    while True:
        time.sleep(24 * 3600)
        print("[BG] 🔄 Rescan Funding quotidien...")
        run_subprocess(SELECT_FUNDING_SCRIPT, "select_funding_pair.py", timeout=600)


def start_background_jobs():
    """
    Démarre les 4 boucles en threads daemon (non-bloquant). Ces threads
    ne font QUE dormir puis lancer un subprocess.run() — ils ne chargent
    eux-mêmes aucune lib lourde (pandas/ccxt/lightgbm), donc leur
    empreinte mémoire propre est négligeable.
    Le régime tourne toujours sur GitHub Actions (Kraken, non bloqué),
    donc pas relancé ici.
    """
    threading.Thread(target=statarb_loop,       daemon=True, name="statarb_thread").start()
    threading.Thread(target=funding_loop,        daemon=True, name="funding_thread").start()
    threading.Thread(target=statarb_rescan_loop, daemon=True, name="statarb_rescan_thread").start()
    threading.Thread(target=funding_rescan_loop, daemon=True, name="funding_rescan_thread").start()
    print("[BG] ✅ Threads de fond lancés (statarb[15min] + funding[8h] + rescans)")
    print("[BG] ℹ️  Régime calculé sur GitHub Actions (cron horaire, Kraken)")