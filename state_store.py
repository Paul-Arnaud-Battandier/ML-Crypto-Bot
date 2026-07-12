"""
state_store.py
────────────────
Remplace les fichiers JSON locaux (current_regime.json, best_pair.json,
funding_state.json, best_funding.json) par une table Supabase unique
'bot_state' (clé/valeur JSON).

Nécessaire car GitHub Actions lance chaque script sur une machine neuve
à chaque fois — aucun fichier local ne survit d'une exécution à l'autre.
Tout l'état doit vivre dans Supabase.

Usage (remplace directement les anciens patterns) :
    from state_store import get_state, set_state

    # Avant : json.load(open('current_regime.json'))
    regime = get_state('current_regime')

    # Avant : json.dump(result, open('current_regime.json', 'w'))
    set_state('current_regime', result)
"""

import os
import requests
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

_SB_URL = os.getenv('SUPABASE_URL', '')
_SB_KEY = os.getenv('SUPABASE_KEY', '')


def get_state(key: str) -> dict | None:
    """
    Lit l'état associé à une clé (ex: 'current_regime').
    Retourne None si absent ou en cas d'erreur — chaque bot doit gérer
    ce cas comme avant (fallback / valeur par défaut / skip).
    """
    if not _SB_URL or not _SB_KEY:
        print("⚠️  SUPABASE_URL/KEY manquants — impossible de lire l'état")
        return None
    try:
        r = requests.get(
            f"{_SB_URL}/rest/v1/bot_state",
            params={'key': f'eq.{key}', 'select': 'value,updated_at'},
            headers={
                'apikey': _SB_KEY,
                'Authorization': f'Bearer {_SB_KEY}',
            },
            timeout=10,
        )
        if r.status_code != 200:
            print(f"⚠️  Erreur lecture état '{key}' : {r.status_code} {r.text[:200]}")
            return None
        rows = r.json()
        if not rows:
            return None
        return rows[0]['value']
    except Exception as e:
        print(f"⚠️  Erreur lecture état '{key}' : {e}")
        return None


def set_state(key: str, value: dict) -> bool:
    """
    Écrit (upsert) l'état associé à une clé.
    Retourne True si succès, False sinon — ne lève jamais d'exception
    pour ne pas casser le bot sur un souci réseau ponctuel.
    """
    if not _SB_URL or not _SB_KEY:
        print("⚠️  SUPABASE_URL/KEY manquants — impossible d'écrire l'état")
        return False
    try:
        r = requests.post(
            f"{_SB_URL}/rest/v1/bot_state",
            params={'on_conflict': 'key'},
            headers={
                'apikey': _SB_KEY,
                'Authorization': f'Bearer {_SB_KEY}',
                'Content-Type': 'application/json',
                'Prefer': 'resolution=merge-duplicates,return=minimal',
            },
            json={'key': key, 'value': value, 'updated_at': datetime.now().isoformat()},
            timeout=10,
        )
        if r.status_code not in (200, 201, 204):
            print(f"⚠️  Erreur écriture état '{key}' : {r.status_code} {r.text[:200]}")
            return False
        return True
    except Exception as e:
        print(f"⚠️  Erreur écriture état '{key}' : {e}")
        return False


def get_state_age_hours(key: str) -> float | None:
    """
    Retourne l'âge en heures du dernier update d'une clé, ou None si absente.
    Remplace la logique _is_stale() qu'on avait dans background_jobs.py.
    """
    if not _SB_URL or not _SB_KEY:
        return None
    try:
        r = requests.get(
            f"{_SB_URL}/rest/v1/bot_state",
            params={'key': f'eq.{key}', 'select': 'updated_at'},
            headers={'apikey': _SB_KEY, 'Authorization': f'Bearer {_SB_KEY}'},
            timeout=10,
        )
        rows = r.json()
        if not rows:
            return None
        updated_at = datetime.fromisoformat(rows[0]['updated_at'].replace('Z', '+00:00'))
        now = datetime.now(updated_at.tzinfo)
        return (now - updated_at).total_seconds() / 3600
    except Exception:
        return None
