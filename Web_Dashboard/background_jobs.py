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
    """Recalcule le régime au démarrage puis toutes les heures.
    Backoff long en cas d'erreur réseau/ban pour ne pas spammer Binance."""
    from compute_regime import get_current_regime  # type: ignore

    # Petit délai aléatoire au démarrage pour éviter les bursts
    # si plusieurs redéploiements se chevauchent
    time.sleep(random.uniform(2, 8))

    print("[BG] 🎯 Regime loop démarré")
    try:
        get_current_regime(verbose=True)
    except Exception as e:
        print(f"[BG] ⚠️ Erreur calcul régime initial : {e}")
        print("[BG] ⏳ Pause 5min avant nouvelle tentative (évite spam API)")
        time.sleep(300)

    while True:
        try:
            now = datetime.now()
            if now.minute == 0 and now.second < 10:
                get_current_regime(verbose=True)
                time.sleep(60)
            else:
                time.sleep(20)
        except Exception as e:
            print(f"[BG] ❌ Erreur regime_loop : {e}")
            print("[BG] ⏳ Pause 5min avant retry (protection rate-limit)")
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