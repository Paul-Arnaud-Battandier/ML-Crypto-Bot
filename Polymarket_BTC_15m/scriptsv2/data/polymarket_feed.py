"""
polymarket_feed.py
------------------
Collecte les données du marché Polymarket BTC up/down 15 minutes.

Approche : slug DÉTERMINISTE calculé mathématiquement.
  - Pas de recherche dans l'API, pas de dépendance à un listing dynamique
  - Slug = btc-updown-15m-{unix_timestamp_début_tranche}
  - Fonctionne toujours, même si le marché vient d'ouvrir

Structure :
  - Gamma API  → récupère les token IDs (YES / NO)
  - CLOB API   → carnet d'ordres (bid/ask/spread/volume)
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API  = "https://clob.polymarket.com"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Slug déterministe — cœur de la logique
# ---------------------------------------------------------------------------
def get_current_15m_slug() -> tuple[str, datetime]:
    """
    Calcule le slug du marché BTC 15m EN COURS sans appel API.

    Exemple : à 14h26 UTC → tranche 14h15 → slug btc-updown-15m-1234567890

    Returns:
        (slug, window_start_utc)
    """
    now          = datetime.now(timezone.utc)
    minute_start = (now.minute // 15) * 15
    window_start = now.replace(minute=minute_start, second=0, microsecond=0)
    timestamp    = int(window_start.timestamp())
    slug         = f"btc-updown-15m-{timestamp}"

    logger.info(f"Tranche courante : {window_start.strftime('%H:%M UTC')} | slug={slug}")
    return slug, window_start


def get_next_15m_slug() -> tuple[str, datetime]:
    """
    Calcule le slug de la PROCHAINE tranche (pour anticiper l'ouverture).
    Utile si on veut entrer au tout début du marché suivant.
    """
    now          = datetime.now(timezone.utc)
    minute_start = (now.minute // 15) * 15
    window_start = now.replace(minute=minute_start, second=0, microsecond=0)
    from datetime import timedelta
    next_window  = window_start + timedelta(minutes=15)
    timestamp    = int(next_window.timestamp())
    slug         = f"btc-updown-15m-{timestamp}"
    return slug, next_window


# ---------------------------------------------------------------------------
# Récupération des token IDs via Gamma API
# ---------------------------------------------------------------------------
def get_token_ids(slug: str) -> Optional[tuple[str, str, dict]]:
    """
    Récupère les token IDs YES et NO pour un slug donné.

    Returns:
        (token_yes, token_no, market_meta) ou None si marché introuvable.
    """
    url = f"{GAMMA_API}/events?slug={slug}"

    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Gamma API error : {e}")
        return None

    events = res.json()
    if not events:
        logger.warning(f"Marché introuvable pour slug={slug}")
        return None

    event   = events[0]
    markets = event.get("markets", [])
    if not markets:
        logger.error("Aucun sous-marché dans l'événement")
        return None

    market      = markets[0]
    clob_tokens = json.loads(market.get("clobTokenIds", "[]"))

    if len(clob_tokens) < 2:
        logger.error(f"Token IDs insuffisants : {clob_tokens}")
        return None

    token_yes = clob_tokens[0]   # YES = BTC monte
    token_no  = clob_tokens[1]   # NO  = BTC baisse

    meta = {
        "event_title" : event.get("title"),
        "market_id"   : market.get("id"),
        "question"    : market.get("question"),
        "end_date"    : market.get("endDate"),
        "condition_id": market.get("conditionId"),
    }

    logger.info(f"Tokens OK | YES={token_yes[:8]}... | NO={token_no[:8]}...")
    return token_yes, token_no, meta


# ---------------------------------------------------------------------------
# Carnet d'ordres via CLOB API
# ---------------------------------------------------------------------------
def get_orderbook(token_id: str, depth: int = 5) -> dict:
    """
    Récupère le carnet d'ordres pour un token.

    Returns:
        dict : {best_bid, best_ask, mid, spread, bid_vol, ask_vol, imbalance}
    """
    empty = {
        "best_bid": None, "best_ask": None, "mid": None,
        "spread": None, "bid_vol": None, "ask_vol": None,
        "imbalance": None,
    }

    try:
        res = requests.get(
            f"{CLOB_API}/book",
            params={"token_id": token_id},
            timeout=10,
        )
        res.raise_for_status()
        book = res.json()
    except requests.RequestException as e:
        logger.warning(f"CLOB error token {token_id[:8]}... : {e}")
        return empty

    bids = book.get("bids", [])
    asks = book.get("asks", [])

    if not bids or not asks:
        logger.warning(f"Orderbook vide pour {token_id[:8]}...")
        return empty

    best_bid = float(bids[0]["price"])
    best_ask = float(asks[0]["price"])
    mid      = (best_bid + best_ask) / 2
    spread   = best_ask - best_bid

    # Volume top-N niveaux
    bid_vol = sum(float(b["size"]) for b in bids[:depth])
    ask_vol = sum(float(a["size"]) for a in asks[:depth])

    # Order imbalance : > 0 = pression achat, < 0 = pression vente
    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0

    return {
        "best_bid" : best_bid,
        "best_ask" : best_ask,
        "mid"      : mid,
        "spread"   : spread,
        "bid_vol"  : bid_vol,
        "ask_vol"  : ask_vol,
        "imbalance": imbalance,
    }


# ---------------------------------------------------------------------------
# Snapshot complet — fonction principale appelée par le bot
# ---------------------------------------------------------------------------
def get_market_snapshot() -> Optional[dict]:
    """
    Snapshot complet du marché BTC 15m courant.

    Returns:
        dict avec tout ce dont on a besoin :
        {
            timestamp, slug, window_start,
            event_title, question, end_date, condition_id,
            # Côté YES (BTC monte)
            yes_mid, yes_bid, yes_ask, yes_spread,
            yes_bid_vol, yes_ask_vol, yes_imbalance,
            # Côté NO (BTC baisse)
            no_mid, no_bid, no_ask, no_spread,
            no_bid_vol, no_ask_vol, no_imbalance,
        }
        ou None si le marché n'est pas encore ouvert.
    """
    slug, window_start = get_current_15m_slug()

    result = get_token_ids(slug)
    if result is None:
        return None
    token_yes, token_no, meta = result

    yes_book = get_orderbook(token_yes)
    no_book  = get_orderbook(token_no)

    snapshot = {
        "timestamp"   : datetime.now(timezone.utc).isoformat(),
        "slug"        : slug,
        "window_start": window_start.isoformat(),
        **meta,
        # YES (UP)
        "yes_mid"      : yes_book["mid"],
        "yes_bid"      : yes_book["best_bid"],
        "yes_ask"      : yes_book["best_ask"],
        "yes_spread"   : yes_book["spread"],
        "yes_bid_vol"  : yes_book["bid_vol"],
        "yes_ask_vol"  : yes_book["ask_vol"],
        "yes_imbalance": yes_book["imbalance"],
        # NO (DOWN)
        "no_mid"       : no_book["mid"],
        "no_bid"       : no_book["best_bid"],
        "no_ask"       : no_book["best_ask"],
        "no_spread"    : no_book["spread"],
        "no_bid_vol"   : no_book["bid_vol"],
        "no_ask_vol"   : no_book["ask_vol"],
        "no_imbalance" : no_book["imbalance"],
    }

    logger.info(
        f"Snapshot OK | YES={snapshot['yes_mid']:.3f} | "
        f"NO={snapshot['no_mid']:.3f} | "
        f"Spread YES={snapshot['yes_spread']:.4f}"
    )
    return snapshot


# ---------------------------------------------------------------------------
# Features pour le meta-model
# ---------------------------------------------------------------------------
def extract_poly_features(snapshot: dict) -> dict:
    """
    Extrait les features Polymarket pour le Random Forest meta-model.

    Features :
      poly_yes_prob    — probabilité implicite marché pour UP
      poly_no_prob     — probabilité implicite marché pour DOWN
      poly_skew        — yes_mid - 0.5 (biais directionnel du marché)
      poly_spread_yes  — spread YES (proxy coût de transaction / liquidité)
      poly_spread_no   — spread NO
      poly_imb_yes     — order imbalance côté YES
      poly_imb_no      — order imbalance côté NO
      poly_net_imb     — imbalance YES - imbalance NO (signal directionnel)
    """
    yes = snapshot.get("yes_mid")
    no  = snapshot.get("no_mid")

    return {
        "poly_yes_prob"   : yes,
        "poly_no_prob"    : no,
        "poly_skew"       : (yes - 0.5) if yes is not None else None,
        "poly_spread_yes" : snapshot.get("yes_spread"),
        "poly_spread_no"  : snapshot.get("no_spread"),
        "poly_imb_yes"    : snapshot.get("yes_imbalance"),
        "poly_imb_no"     : snapshot.get("no_imbalance"),
        "poly_net_imb"    : (
            snapshot["yes_imbalance"] - snapshot["no_imbalance"]
            if snapshot.get("yes_imbalance") is not None
            and snapshot.get("no_imbalance") is not None
            else None
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    print("=== Test slug déterministe ===")
    slug, window = get_current_15m_slug()
    print(f"Slug    : {slug}")
    print(f"Tranche : {window.strftime('%Y-%m-%d %H:%M UTC')}")

    print("\n=== Snapshot marché ===")
    snap = get_market_snapshot()
    if snap:
        import pprint
        pprint.pprint(snap)
        print("\n=== Features meta-model ===")
        pprint.pprint(extract_poly_features(snap))
    else:
        print("Marché non disponible (peut-être entre deux tranches).")