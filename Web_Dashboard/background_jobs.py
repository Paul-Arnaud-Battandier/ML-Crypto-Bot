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

ROOT_DIR = Path(__file__).parent.parent  # ML_Crypto_Bot/
sys.path.insert(0, str(ROOT_DIR))
from state_store import get_state, set_state

# ── Chemins vers les scripts ────────────────────────────────────
STATARB_SCRIPT        = ROOT_DIR / "StatArb_ETH_15m"    / "scripts" / "live_bot.py"
FUNDING_SCRIPT         = ROOT_DIR / "FundingCarry_Multi" / "scripts" / "live_funding_bot.py"
SELECT_PAIR_SCRIPT     = ROOT_DIR / "Regime_Detector"    / "scripts" / "select_pair.py"
SELECT_FUNDING_SCRIPT  = ROOT_DIR / "FundingCarry_Multi" / "scripts" / "select_funding_pair.py"

# Verrou global : empêche DEUX subprocess de tourner en même temps, peu
# importe le calage horaire. Le crash du 01/09 est arrivé parce que le
# rescan funding (calé arbitrairement "24h après le démarrage du thread")
# est retombé pile sur un scan StatArb (toutes les 15min) — un simple
# décalage d'horaire ne suffit pas à garantir que ça ne se reproduise
# jamais après un futur redéploiement à une heure différente. Ce verrou,
# lui, le garantit dans tous les cas : si un subprocess tourne déjà, le
# suivant attend qu'il se termine avant de démarrer.
# Verrou LOCAL : empêche deux subprocess de tourner en même temps dans CE
# process. Le crash du 01/09 est arrivé parce que le rescan funding (calé
# arbitrairement "24h après le démarrage du thread") est retombé pile sur
# un scan StatArb (toutes les 15min).
_subprocess_lock = threading.Lock()

# Verrou DISTRIBUÉ (Supabase) : le verrou local ci-dessus ne protège que
# dans un seul process Python. Render fait des déploiements zero-downtime
# — l'ancienne et la nouvelle instance tournent parfois en parallèle
# quelques dizaines de secondes, chacune avec SON PROPRE verrou local
# indépendant. Ça a très probablement causé le double-cycle funding
# constaté le 04/09 (deux runs à 19 secondes d'écart). Ce verrou-ci vit
# dans Supabase, donc il est partagé même entre deux containers distincts.
# Best-effort (pas de compare-and-swap atomique via ce simple REST) — ça
# réduit la fenêtre de race de la durée d'un subprocess entier (10-60s) à
# la fraction de seconde entre lecture et écriture. Suffisant ici : on est
# en paper trading, le pire cas est une double-position en démo, pas une
# perte réelle.
DIST_LOCK_KEY         = "subprocess_lock"
DIST_LOCK_TTL_SECONDS = 300   # expire après 5min — largement au-dessus de
                               # la durée normale d'un subprocess, pour ne
                               # jamais rester bloqué si un run crash sans
                               # relâcher le verrou.
DIST_LOCK_MAX_WAIT    = 30    # temps max d'attente avant de forcer le
                               # passage (mieux qu'un blocage indéfini)


def _acquire_distributed_lock(label, poll_interval=2):
    waited = 0
    while waited < DIST_LOCK_MAX_WAIT:
        lock = get_state(DIST_LOCK_KEY)
        if lock is None:
            break
        try:
            locked_at = datetime.fromisoformat(lock['locked_at'])
            age = (datetime.now(timezone.utc) - locked_at).total_seconds()
        except Exception:
            age = DIST_LOCK_TTL_SECONDS + 1  # lock corrompu → considéré expiré
        if age > DIST_LOCK_TTL_SECONDS:
            print(f"[BG] ⚠️ Verrou distribué expiré (posé par '{lock.get('locked_by')}', "
                  f"{age:.0f}s) — probablement un crash sans libération. On le reprend.")
            break
        time.sleep(poll_interval)
        waited += poll_interval
    else:
        print(f"[BG] ⚠️ Verrou distribué toujours pris après {DIST_LOCK_MAX_WAIT}s "
              f"— on lance quand même {label} (mieux qu'un blocage indéfini).")

    set_state(DIST_LOCK_KEY, {
        'locked_by': label,
        'locked_at': datetime.now(timezone.utc).isoformat(),
    })


def _release_distributed_lock():
    set_state(DIST_LOCK_KEY, None)


def run_subprocess(script_path, label, timeout=280):
    """
    Lance un script en sous-processus isolé (StatArb, Funding, ou un
    rescan). Toute la mémoire utilisée est rendue à l'OS dès que le
    sous-processus se termine — jamais résidente dans le process
    principal Flask.
    Double verrou : local (ce process) + distribué (Supabase, protège
    aussi entre deux containers Render distincts pendant un déploiement).
    """
    with _subprocess_lock:
        _acquire_distributed_lock(label)
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
        finally:
            _release_distributed_lock()


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


def _seconds_until_next_daily_time(hour, minute, buffer_seconds=5):
    """Secondes jusqu'au prochain HH:MM local (aujourd'hui ou demain)."""
    from datetime import timedelta
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds() + buffer_seconds


def _seconds_until_next_weekly_time(weekday, hour, minute, buffer_seconds=5):
    """Secondes jusqu'au prochain jour de semaine (0=lundi) à HH:MM local."""
    from datetime import timedelta
    now = datetime.now()
    days_ahead = (weekday - now.weekday()) % 7
    target = (now + timedelta(days=days_ahead)).replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=7)
    return (target - now).total_seconds() + buffer_seconds


def statarb_rescan_loop():
    """Relance le scan de cointégration chaque lundi à 04:07 (heure fixe,
    creuse, décalée de 15min pour ne jamais tomber pile sur un scan
    StatArb même si le timing dérive légèrement)."""
    while True:
        wait = _seconds_until_next_weekly_time(weekday=0, hour=4, minute=7)
        time.sleep(wait)
        print("[BG] 🔄 Rescan StatArb hebdomadaire...")
        run_subprocess(SELECT_PAIR_SCRIPT, "select_pair.py", timeout=600)


def funding_rescan_loop():
    """Relance le scan de sélection funding chaque jour à 04:22 (heure fixe,
    creuse, décalée du rescan StatArb pour éviter tout chevauchement)."""
    while True:
        wait = _seconds_until_next_daily_time(hour=4, minute=22)
        time.sleep(wait)
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