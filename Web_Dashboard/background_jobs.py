"""
background_jobs.py
───────────────────
Lance live_regime.py et live_bot.py en arrière-plan (threads daemon)
à l'intérieur du process Flask, pour que tout tourne dans le même
service Render (gratuit) au lieu d'un Background Worker payant.

⚠️ IMPORTANT : ne JAMAIS lancer live_bot.py en local en même temps
que ce service tourne sur Render — ça créerait deux bots qui tradent
en parallèle sur le même compte Binance.
"""

import sys
import time
import random
import threading
from pathlib import Path
from datetime import datetime

# ── Chemins vers les modules des autres dossiers ───────────────
ROOT_DIR = Path(__file__).parent.parent  # ML_Crypto_Bot/
sys.path.insert(0, str(ROOT_DIR / "Regime_Detector" / "scripts"))
sys.path.insert(0, str(ROOT_DIR / "StatArb_ETH_15m"  / "scripts"))


def regime_loop():
    """
    Recalcule le régime au démarrage puis toutes les heures.
    Utilise un timer basé sur le temps écoulé (pas une fenêtre
    d'horloge exacte) pour éviter de rater le déclenchement à
    cause d'un léger décalage de timing entre les checks.
    """
    from compute_regime import get_current_regime  # type: ignore

    # Petit délai aléatoire au démarrage pour éviter les bursts
    # si plusieurs redéploiements se chevauchent
    time.sleep(random.uniform(2, 8))

    print("[BG] 🎯 Regime loop démarré")

    REFRESH_SECONDS = 3600  # 1h
    last_run = None

    def try_run():
        nonlocal last_run
        try:
            get_current_regime(verbose=True)
            last_run = datetime.now()
            return True
        except Exception as e:
            print(f"[BG] ⚠️ Erreur calcul régime : {e}")
            return False

    # Premier calcul immédiat
    if not try_run():
        print("[BG] ⏳ Pause 5min avant nouvelle tentative (évite spam API)")
        time.sleep(300)
        try_run()  # Si échec encore, last_run reste None → retry rapide via boucle ci-dessous

    while True:
        try:
            elapsed = (datetime.now() - last_run).total_seconds() if last_run else REFRESH_SECONDS
            if elapsed >= REFRESH_SECONDS:
                if not try_run():
                    print("[BG] ⏳ Pause 5min avant retry (protection rate-limit)")
                    time.sleep(300)
                    continue
            time.sleep(30)  # check toutes les 30s si on a dépassé le délai
        except Exception as e:
            print(f"[BG] ❌ Erreur regime_loop : {e}")
            time.sleep(300)


def trading_loop():
    """Lance live_bot.py en continu, avec backoff exponentiel si crash."""
    from live_bot import main as bot_main  # type: ignore

    print("[BG] 🤖 Trading loop démarré")
    backoff = 30  # secondes, double à chaque crash, plafonné à 10min

    while True:
        try:
            bot_main()  # Contient déjà sa propre boucle infinie
            backoff = 30  # reset si jamais main() retourne proprement
        except Exception as e:
            print(f"[BG] ❌ Bot crashé : {e}")
            print(f"[BG] ⏳ Relance dans {backoff}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 600)  # cap à 10 minutes


def start_background_jobs():
    """Démarre les deux boucles en threads daemon (non-bloquant)."""
    threading.Thread(target=regime_loop,  daemon=True, name="regime_thread").start()
    threading.Thread(target=trading_loop, daemon=True, name="trading_thread").start()
    print("[BG] ✅ Threads de fond lancés (régime + trading)")