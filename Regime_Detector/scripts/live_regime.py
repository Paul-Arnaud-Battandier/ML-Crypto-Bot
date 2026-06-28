"""
live_regime.py
──────────────
Boucle toutes les heures, recalcule le régime et
met à jour current_regime.json.
À lancer en parallèle des bots de trading.
"""

import time
from datetime import datetime
from compute_regime import get_current_regime

def main():
    print("="*50)
    print("🎯 RÉGIME DETECTOR EN LIGNE")
    print("   Mise à jour toutes les heures")
    print("="*50)

    while True:
        try:
            now = datetime.now()

            # Calcul au début de chaque heure
            if now.minute == 0 and now.second < 10:
                print(f"\n[{now.strftime('%H:%M:%S')}] 🔄 Mise à jour du régime...")
                result = get_current_regime(verbose=True)

                # Alerte si changement de régime
                # (comparaison avec le précédent dans une prochaine version)

                time.sleep(60)  # Évite double-calcul dans la même minute

            else:
                time.sleep(1)

        except Exception as e:
            print(f"❌ Erreur : {e}")
            time.sleep(10)

if __name__ == "__main__":
    # Premier calcul immédiat au démarrage
    print("🚀 Calcul initial...")
    get_current_regime(verbose=True)
    print("\n⏳ En attente du prochain cycle horaire...")
    main()