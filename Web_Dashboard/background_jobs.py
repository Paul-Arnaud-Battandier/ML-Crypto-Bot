"""
background_jobs.py
───────────────────
Lance live_regime.py et live_bot.py en arrière-plan (threads daemon)
à l'intérieur du process Flask, pour que tout tourne dans le même
service Render (gratuit) au lieu d'un Background Worker payant.

⚠️ IMPORTANT : ne JAMAIS lancer live_bot.py en local en même temps
que ce service tourne sur Render — ça créerait deux bots qui tradent
en parallèle sur le même compte Binance.

── Note mémoire ─────────────────────────────────────────────────
Les scans de sélection de paire (select_pair.py, select_funding_pair.py)
importent statsmodels/scipy (cointégration, OLS). Si on les importe
directement dans ce process (comme avant), ces librairies restent
chargées en RAM pour toujours — Python ne décharge jamais un module
importé, même si le scan ne tourne qu'1x/semaine.

On les lance donc en SOUS-PROCESSUS (subprocess.run). Un sous-processus
a sa propre mémoire ; quand il se termine, tout est rendu à l'OS —
contrairement à un import qui reste résident dans le process principal.
"""

import sys
import time
import random
import threading
import gc
import subprocess
from pathlib import Path
from datetime import datetime

# ── Chemins vers les modules des autres dossiers ───────────────
ROOT_DIR = Path(__file__).parent.parent  # ML_Crypto_Bot/
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "Regime_Detector"    / "scripts"))
sys.path.insert(0, str(ROOT_DIR / "StatArb_ETH_15m"    / "scripts"))
sys.path.insert(0, str(ROOT_DIR / "FundingCarry_Multi" / "scripts"))

SELECT_PAIR_SCRIPT    = ROOT_DIR / "Regime_Detector"    / "scripts" / "select_pair.py"
SELECT_FUNDING_SCRIPT = ROOT_DIR / "FundingCarry_Multi" / "scripts" / "select_funding_pair.py"


def run_scan_subprocess(script_path, label, timeout=600):
    """
    Lance un script de scan (select_pair.py / select_funding_pair.py) en
    sous-processus isolé. Toute la mémoire utilisée (statsmodels, scipy,
    DataFrames temporaires) est rendue à l'OS dès que le sous-processus
    se termine — elle ne reste jamais résidente dans le process principal.
    """
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, timeout=timeout,
        )
        # On relaie la sortie du script dans les logs Render
        if result.stdout:
            print(result.stdout)
        if result.returncode != 0:
            print(f"[BG] ⚠️ {label} a terminé avec le code {result.returncode}")
            if result.stderr:
                print(result.stderr[-1500:])  # dernières lignes utiles
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"[BG] ⚠️ {label} a dépassé le timeout ({timeout}s)")
        return False
    except Exception as e:
        print(f"[BG] ⚠️ Erreur lancement {label} : {e}")
        return False


def _is_stale(state_key, max_age_hours):
    """
    Retourne True si l'état Supabase (bot_state[state_key]) est absent
    ou dépasse max_age_hours. Remplace l'ancien check basé sur des
    fichiers locaux — ceux-ci ne survivent pas aux redéploiements Render
    (filesystem éphémère), donc lire un fichier local juste après un
    push renvoyait toujours "absent" et forçait un rescan inutile à
    chaque déploiement.
    """
    from state_store import get_state_age_hours
    age = get_state_age_hours(state_key)
    if age is None:
        return True
    return age > max_age_hours


def statarb_rescan_loop():
    """Relance le scan de cointégration (11 paires candidates) toutes les 7
    jours, en sous-processus isolé — libère statsmodels/scipy après coup."""
    while True:
        time.sleep(7 * 24 * 3600)  # 7 jours
        print("[BG] 🔄 Rescan StatArb hebdomadaire...")
        run_scan_subprocess(SELECT_PAIR_SCRIPT, "select_pair.py")
        gc.collect()


def trading_loop():
    """Lance select_pair.py (scan uniquement si périmé, en sous-processus),
    puis live_bot.py en continu, avec backoff exponentiel si crash."""
    from live_bot import main as bot_main  # type: ignore
    from config import PATHS  # type: ignore

    # Démarre en 1er (des 2 boucles restantes) — décalé pour lisser la charge API
    time.sleep(random.uniform(30, 45))

    print("[BG] 🤖 Trading loop démarré")

    # Ne rescanner que si best_pair.json est absent ou vieux de +7 jours —
    # évite de reprovoquer un ban Binance à chaque redéploiement Render.
    if _is_stale('best_pair', max_age_hours=7*24):
        run_scan_subprocess(SELECT_PAIR_SCRIPT, "select_pair.py")
        gc.collect()
    else:
        print("[BG] 📊 best_pair.json récent — scan initial sauté")

    # Thread de rescan hebdomadaire (léger — ne fait que dormir + subprocess)
    threading.Thread(target=statarb_rescan_loop, daemon=True, name="statarb_rescan_thread").start()

    backoff = 30  # secondes, double à chaque crash, plafonné à 10min

    while True:
        try:
            bot_main()  # Contient déjà sa propre boucle infinie
            backoff = 30
        except Exception as e:
            print(f"[BG] ❌ Bot crashé : {e}")
            print(f"[BG] ⏳ Relance dans {backoff}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 600)


def funding_rescan_loop():
    """Relance le scan de sélection funding toutes les 24h, en sous-processus
    isolé — même logique que statarb_rescan_loop."""
    while True:
        time.sleep(24 * 3600)  # 24h
        print("[BG] 🔄 Rescan Funding quotidien...")
        run_scan_subprocess(SELECT_FUNDING_SCRIPT, "select_funding_pair.py")
        gc.collect()


def funding_loop():
    """Lance select_funding_pair.py (scan uniquement si périmé, en
    sous-processus) puis live_funding_bot.py en continu."""
    from live_funding_bot import main as funding_main  # type: ignore
    from config import PATHS  # type: ignore

    # Démarre en 3ème — décalé pour laisser respirer les 2 autres bots
    time.sleep(random.uniform(90, 120))
    print("[BG] 💰 Funding loop démarré")

    if _is_stale('best_funding', max_age_hours=24):
        run_scan_subprocess(SELECT_FUNDING_SCRIPT, "select_funding_pair.py")
        gc.collect()
    else:
        print("[BG] 💰 best_funding.json récent — scan initial sauté")

    threading.Thread(target=funding_rescan_loop, daemon=True, name="funding_rescan_thread").start()

    backoff = 30
    while True:
        try:
            funding_main()
            backoff = 30
        except Exception as e:
            print(f"[BG] ❌ Funding bot crashé : {e}")
            print(f"[BG] ⏳ Relance dans {backoff}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 600)


def start_background_jobs():
    """Démarre les deux boucles principales en threads daemon (non-bloquant).
    Le régime tourne maintenant sur GitHub Actions (cron horaire, Kraken —
    Binance bloque les IP de datacenter US) et écrit dans Supabase, plus
    besoin de le faire tourner ici. Les scans de sélection de paire
    (StatArb hebdo, Funding quotidien) tournent en sous-processus isolés
    pour ne jamais charger statsmodels/scipy dans la mémoire résidente
    du process principal.
    """
    threading.Thread(target=trading_loop, daemon=True, name="trading_thread").start()
    threading.Thread(target=funding_loop, daemon=True, name="funding_thread").start()
    print("[BG] ✅ Threads de fond lancés (statarb[+rescan 7j] + funding[+rescan 24h])")
    print("[BG] ℹ️  Régime calculé désormais sur GitHub Actions (cron horaire)")