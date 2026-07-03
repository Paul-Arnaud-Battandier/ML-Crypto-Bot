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
import gc
from pathlib import Path
from datetime import datetime

# ── Chemins vers les modules des autres dossiers ───────────────
ROOT_DIR = Path(__file__).parent.parent  # ML_Crypto_Bot/
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "Regime_Detector"    / "scripts"))
sys.path.insert(0, str(ROOT_DIR / "StatArb_ETH_15m"    / "scripts"))
sys.path.insert(0, str(ROOT_DIR / "FundingCarry_Multi" / "scripts"))


def regime_loop():
    """
    Recalcule le régime au démarrage puis toutes les heures.
    Utilise un timer basé sur le temps écoulé (pas une fenêtre
    d'horloge exacte) pour éviter de rater le déclenchement à
    cause d'un léger décalage de timing entre les checks.
    """
    from compute_regime import get_current_regime  # type: ignore

    # Démarre en premier — délai court
    time.sleep(random.uniform(3, 8))

    print("[BG] 🎯 Regime loop démarré")

    REFRESH_SECONDS = 3600  # 1h
    last_run = None

    def try_run():
        nonlocal last_run
        try:
            get_current_regime(verbose=True)
            last_run = datetime.now()
            gc.collect()
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


def _is_stale(json_path, max_age_hours):
    """Retourne True si le fichier n'existe pas ou dépasse max_age_hours."""
    import json as _json
    from datetime import datetime as _dt
    try:
        with open(json_path) as f:
            data = _json.load(f)
        ts_str = data.get('updated') or data.get('timestamp')
        if not ts_str:
            return True
        ts = _dt.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
        age_hours = (_dt.now() - ts).total_seconds() / 3600
        return age_hours > max_age_hours
    except FileNotFoundError:
        return True
    except Exception:
        return True


def statarb_rescan_loop():
    """Relance le scan de cointégration (11 paires candidates) toutes les 7
    jours (thread séparé, tourne en parallèle du bot de trading StatArb)."""
    from select_pair import get_best_pair  # type: ignore

    while True:
        try:
            get_best_pair(verbose=True)
        except Exception as e:
            print(f"[BG] ⚠️ Erreur rescan StatArb pair : {e}")
        time.sleep(7 * 24 * 3600)  # 7 jours


def trading_loop():
    """Lance select_pair.py (scan uniquement si périmé, puis rescan hebdo
    en thread interne) puis live_bot.py en continu, avec backoff si crash."""
    from select_pair import get_best_pair  # type: ignore
    from live_bot import main as bot_main  # type: ignore
    from config import PATHS  # type: ignore

    # Démarre en 2ème — décalé pour ne pas cumuler avec regime_loop
    time.sleep(random.uniform(30, 45))

    print("[BG] 🤖 Trading loop démarré")

    # Ne rescanner que si best_pair.json est absent ou vieux de +7 jours —
    # évite de reprovoquer un ban Binance à chaque redéploiement Render.
    if _is_stale(PATHS['best_pair_json'], max_age_hours=7*24):
        try:
            get_best_pair(verbose=True)
        except Exception as e:
            print(f"[BG] ⚠️ Erreur scan StatArb pair initial : {e}")
    else:
        print("[BG] 📊 best_pair.json récent — scan initial sauté")

    # Thread interne pour le rescan hebdomadaire (indépendant du bot bloquant)
    threading.Thread(target=statarb_rescan_loop, daemon=True, name="statarb_rescan_thread").start()

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


def funding_rescan_loop():
    """Relance le scan de sélection de paire toutes les 24h (thread séparé,
    tourne en parallèle du bot de trading funding)."""
    from select_funding_pair import get_best_funding  # type: ignore

    while True:
        try:
            get_best_funding(verbose=True)
        except Exception as e:
            print(f"[BG] ⚠️ Erreur rescan funding : {e}")
        time.sleep(24 * 3600)  # 24h


def funding_loop():
    """Lance select_funding_pair.py (scan uniquement si périmé, puis rescan
    quotidien en thread interne) puis live_funding_bot.py en continu."""
    from select_funding_pair import get_best_funding  # type: ignore
    from live_funding_bot import main as funding_main   # type: ignore
    from config import PATHS  # type: ignore

    # Démarre en 3ème — décalé pour laisser respirer les 2 autres bots
    time.sleep(random.uniform(90, 120))
    print("[BG] 💰 Funding loop démarré")

    if _is_stale(PATHS['best_funding_json'], max_age_hours=24):
        try:
            get_best_funding(verbose=True)
        except Exception as e:
            print(f"[BG] ⚠️ Erreur scan funding initial : {e}")
    else:
        print("[BG] 💰 best_funding.json récent — scan initial sauté")

    # Thread interne pour le rescan quotidien (indépendant du bot bloquant)
    threading.Thread(target=funding_rescan_loop, daemon=True, name="funding_rescan_thread").start()

    backoff = 30
    while True:
        try:
            funding_main()  # Contient sa propre boucle infinie (check 8h)
            backoff = 30
        except Exception as e:
            print(f"[BG] ❌ Funding bot crashé : {e}")
            print(f"[BG] ⏳ Relance dans {backoff}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 600)


def start_background_jobs():
    """Démarre les trois boucles principales en threads daemon (non-bloquant).
    Chacune lance elle-même un thread de rescan interne pour rester à jour :
      - regime_loop    : recalcul horaire
      - trading_loop    : bot StatArb + rescan de paire hebdomadaire
      - funding_loop    : bot Funding Carry + rescan de paire quotidien
    """
    threading.Thread(target=regime_loop,  daemon=True, name="regime_thread").start()
    threading.Thread(target=trading_loop, daemon=True, name="trading_thread").start()
    threading.Thread(target=funding_loop, daemon=True, name="funding_thread").start()
    print("[BG] ✅ Threads de fond lancés (régime + statarb[+rescan 7j] + funding[+rescan 24h])")