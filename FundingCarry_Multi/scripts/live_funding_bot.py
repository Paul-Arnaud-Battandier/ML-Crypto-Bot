"""
live_funding_bot.py
────────────────────
Funding Rate Carry Bot — Long Spot + Short Perpetual

Logique :
  - Tourne toutes les 8h (aux horaires de paiement UTC : 00:00, 08:00, 16:00)
  - Entrée : funding valid + régime HIGH_VOL ou NEUTRAL + pas de position
  - Sortie  : rolling moyenne des N derniers paiements < seuil négatif
              OU dépassement du MAX_HOLDING_DAYS
  - Profit  = funding collecté (enregistré à chaque paiement) - frais round-trip

Différences clés vs StatArb :
  - Horizon de semaines, pas d'heures
  - Pas de ML — pure règle statistique
  - Check toutes les 8h, pas toutes les 15min
"""

import ccxt
import pandas as pd
import numpy as np
import json
import csv
import os
import sys
import time
import requests as _requests
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# ── Chemins ───────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent.parent   # ML_Crypto_Bot/
sys.path.insert(0, str(ROOT_DIR))
from config import PATHS
from state_store import get_state

CARRY_DIR  = ROOT_DIR / "FundingCarry_Multi" / "data"
CARRY_DIR.mkdir(exist_ok=True)

EQUITY_CSV = CARRY_DIR / "funding_equity.csv"
TRADES_CSV = CARRY_DIR / "funding_trades.csv"
STATE_JSON = CARRY_DIR / "funding_state.json"  # Persistance position entre redémarrages

REGIME_JSON   = PATHS['regime_json']
FUNDING_JSON  = ROOT_DIR / "FundingCarry_Multi" / "data" / ".." / ".." / \
                "FundingCarry_Multi" / "data" / "best_funding.json"
FUNDING_JSON  = ROOT_DIR / "FundingCarry_Multi" / "data" / "best_funding.json"

# ── Paramètres ────────────────────────────────────────────────
TRADE_AMOUNT_USD     = 200.0    # $ par patte (200$ spot + 200$ perp = 400$ total)
BNB_DISCOUNT         = False    # Activer si BNB activé sur le compte
SPOT_FEE             = 0.0010
PERP_FEE             = 0.0005
ROUND_TRIP_FEE       = (SPOT_FEE + PERP_FEE) * 2 * (0.75 if BNB_DISCOUNT else 1.0)

# Sortie : si la moyenne des N derniers paiements est sous ce seuil
EXIT_ROLLING_PAYMENTS = 5       # Nb de paiements dans la fenêtre glissante
EXIT_THRESHOLD_PCT    = -0.005  # % par paiement → sortie si rolling avg < -0.005%
MAX_HOLDING_DAYS      = 45      # Forcer la sortie après 45 jours

# Supabase (optionnel — silencieux si absent)
_SB_URL = os.getenv('SUPABASE_URL', '')
_SB_KEY = os.getenv('SUPABASE_KEY', '')

# ── API Keys ──────────────────────────────────────────────────
API_KEY    = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')


# ── Supabase logging ──────────────────────────────────────────
def supabase_insert(table, data):
    if not _SB_URL or not _SB_KEY:
        return
    try:
        _requests.post(
            f"{_SB_URL}/rest/v1/{table}",
            headers={
                'apikey': _SB_KEY,
                'Authorization': f'Bearer {_SB_KEY}',
                'Content-Type': 'application/json',
                'Prefer': 'return=minimal',
            },
            json=data, timeout=3,
        )
    except:
        pass


# ── CSV logging ───────────────────────────────────────────────
def init_csv():
    if not EQUITY_CSV.exists():
        with open(EQUITY_CSV, 'w', newline='') as f:
            csv.writer(f).writerow([
                'timestamp', 'equity_usdt', 'position_status',
                'symbol', 'funding_collected_usd', 'unrealized_pnl_usd',
            ])
    if not TRADES_CSV.exists():
        with open(TRADES_CSV, 'w', newline='') as f:
            csv.writer(f).writerow([
                'timestamp', 'action', 'symbol', 'spot_price', 'perp_price',
                'basis_pct', 'funding_rate_pct', 'funding_apr',
                'size_usd', 'exit_reason', 'total_funding_collected_usd',
            ])

def log_equity(equity, position, symbol, funding_collected, unrealized_pnl):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    row = [ts, round(equity, 4), position, symbol or '',
           round(funding_collected, 4), round(unrealized_pnl, 4)]
    with open(EQUITY_CSV, 'a', newline='') as f:
        csv.writer(f).writerow(row)
    supabase_insert('funding_equity', {
        'timestamp'             : ts,
        'equity_usdt'           : round(equity, 4),
        'position_status'       : position,
        'symbol'                : symbol or '',
        'funding_collected_usd' : round(funding_collected, 4),
        'unrealized_pnl_usd'    : round(unrealized_pnl, 4),
    })

def log_trade(action, symbol, spot_price, perp_price, funding_rate,
              exit_reason=None, total_funding=None):
    basis_pct = (perp_price - spot_price) / spot_price * 100 if spot_price else 0
    apr = funding_rate * 3 * 365 * 100
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(TRADES_CSV, 'a', newline='') as f:
        csv.writer(f).writerow([
            ts, action, symbol,
            round(spot_price, 4), round(perp_price, 4),
            round(basis_pct, 4), round(funding_rate * 100, 5),
            round(apr, 2), TRADE_AMOUNT_USD,
            exit_reason or '', round(total_funding or 0, 4),
        ])
    supabase_insert('funding_trades', {
        'timestamp'                   : ts,
        'action'                      : action,
        'symbol'                      : symbol,
        'spot_price'                  : round(spot_price, 4),
        'perp_price'                  : round(perp_price, 4),
        'basis_pct'                   : round(basis_pct, 4),
        'funding_rate_pct'            : round(funding_rate * 100, 5),
        'funding_apr'                 : round(apr, 2),
        'size_usd'                    : TRADE_AMOUNT_USD,
        'exit_reason'                 : exit_reason,
        'total_funding_collected_usd' : round(total_funding or 0, 4),
    })


# ── Persistance de la position (survie aux redémarrages) ──────
def save_state(state: dict):
    with open(STATE_JSON, 'w') as f:
        json.dump(state, f, indent=2)

def load_state() -> dict:
    try:
        with open(STATE_JSON) as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            'position': 'FLAT',
            'symbol': None,
            'entry_time': None,
            'entry_spot_price': None,
            'entry_perp_price': None,
            'spot_size': None,
            'perp_size': None,
            'funding_payments': [],   # liste des funding rates collectés
            'funding_collected_usd': 0.0,
        }


# ── Exchanges ─────────────────────────────────────────────────
def init_exchanges():
    spot = ccxt.binance({
        'apiKey': API_KEY, 'secret': API_SECRET,
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    perp = ccxt.binance({
        'apiKey': API_KEY, 'secret': API_SECRET,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    spot.enable_demo_trading(True)
    perp.enable_demo_trading(True)
    return spot, perp


# ── Données de marché ─────────────────────────────────────────
def get_prices(spot_ex, perp_ex, symbol):
    spot_ticker = spot_ex.fetch_ticker(symbol)
    perp_ticker = perp_ex.fetch_ticker(symbol)
    return float(spot_ticker['last']), float(perp_ticker['last'])

def get_current_funding_rate(perp_ex, symbol):
    fr = perp_ex.fetch_funding_rate(symbol)
    return float(fr.get('fundingRate', 0))


# ── Lectures régime et paire ──────────────────────────────────
def read_regime():
    """
    Lit le régime actuel depuis Supabase (bot_state['current_regime']).
    compute_regime.py tourne maintenant sur GitHub Actions (Kraken),
    donc plus de fichier local à lire — voir live_bot.py pour le détail.
    """
    return get_state('current_regime')

def read_best_funding():
    """Lit la meilleure opportunité depuis Supabase (bot_state['best_funding'])"""
    return get_state('best_funding')


# ── Logique de sortie ─────────────────────────────────────────
def should_exit(state, current_funding_rate) -> tuple[bool, str]:
    """
    Retourne (True, raison) si on doit sortir, sinon (False, '').
    Deux conditions de sortie :
      1. Rolling moyenne des derniers paiements trop négative
      2. Durée max dépassée
    """
    # Durée max
    if state.get('entry_time'):
        entry_dt = datetime.fromisoformat(state['entry_time'])
        held_days = (datetime.now() - entry_dt).days
        if held_days >= MAX_HOLDING_DAYS:
            return True, f"TIME_LIMIT ({held_days}j)"

    # Rolling average des N derniers paiements
    payments = state.get('funding_payments', []) + [current_funding_rate * 100]
    if len(payments) >= EXIT_ROLLING_PAYMENTS:
        rolling_avg = np.mean(payments[-EXIT_ROLLING_PAYMENTS:])
        if rolling_avg < EXIT_THRESHOLD_PCT:
            return True, f"FUNDING_NEGATIVE (avg={rolling_avg:.4f}%)"

    return False, ''


# ── Exécution des ordres ──────────────────────────────────────
def enter_position(spot_ex, perp_ex, symbol, spot_price, perp_price):
    """Long Spot + Short Perp"""
    spot_size = round(TRADE_AMOUNT_USD / spot_price, 4)
    perp_size = round(TRADE_AMOUNT_USD / perp_price, 3)

    print(f"  💸 Entrée : Long {spot_size} {symbol} Spot @ ${spot_price:.2f}")
    print(f"  💸 Entrée : Short {perp_size} {symbol} Perp @ ${perp_price:.2f}")

    try:
        spot_ex.create_market_buy_order(symbol, spot_size)
    except Exception as e:
        print(f"  ❌ Échec ordre spot buy : {e}")
        return None

    try:
        perp_ex.create_market_sell_order(symbol, perp_size)
    except Exception as e:
        print(f"  ❌ Échec ordre perp sell : {e} — annulation spot")
        try:
            spot_ex.create_market_sell_order(symbol, spot_size)
        except Exception as e2:
            print(f"  🚨 ALERTE : impossible d'annuler le spot : {e2}")
        return None

    return spot_size, perp_size

def exit_position(spot_ex, perp_ex, state):
    """Sell Spot + Buy Perp"""
    symbol    = state['symbol']
    spot_size = state['spot_size']
    perp_size = state['perp_size']

    print(f"  💸 Sortie : Sell {spot_size} {symbol} Spot")
    print(f"  💸 Sortie : Buy  {perp_size} {symbol} Perp")

    for attempt in range(2):
        try:
            spot_ex.create_market_sell_order(symbol, spot_size)
            break
        except Exception as e:
            if attempt == 0:
                print(f"  ❌ Échec spot sell — retry dans 10s : {e}")
                time.sleep(10)
            else:
                print(f"  🚨 ALERTE : impossible de fermer spot : {e}")

    for attempt in range(2):
        try:
            perp_ex.create_market_buy_order(symbol, perp_size)
            break
        except Exception as e:
            if attempt == 0:
                print(f"  ❌ Échec perp buy — retry dans 10s : {e}")
                time.sleep(10)
            else:
                print(f"  🚨 ALERTE : impossible de fermer perp : {e}")


# ── Calcul PnL mark-to-market ─────────────────────────────────
def compute_unrealized_pnl(state, current_spot, current_perp):
    if state['position'] == 'FLAT':
        return 0.0
    spot_pnl = (current_spot - state['entry_spot_price']) * state['spot_size']
    perp_pnl = (state['entry_perp_price'] - current_perp) * state['perp_size']
    return spot_pnl + perp_pnl  # USD


# ── Main ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("💰 FUNDING RATE CARRY BOT (BINANCE DEMO)")
    print(f"   Capital/patte : ${TRADE_AMOUNT_USD} | Frais RT : {ROUND_TRIP_FEE*100:.3f}%")
    print(f"   Exit trigger  : rolling {EXIT_ROLLING_PAYMENTS} paiements < {EXIT_THRESHOLD_PCT}%")
    print(f"   Max holding   : {MAX_HOLDING_DAYS} jours")
    print("=" * 60)

    init_csv()
    spot_ex, perp_ex = init_exchanges()
    state = load_state()

    print(f"✅ API connectée")
    print(f"📊 Position actuelle : {state['position']}"
          f"{(' — ' + state['symbol']) if state['symbol'] else ''}")

    while True:
        try:
            now = datetime.now()
            # Check toutes les 8h (00, 08, 16 UTC) + 2 minutes de décalage
            # pour être sûr que le paiement a été effectué
            utc_hour = datetime.utcnow().hour
            if utc_hour in (0, 8, 16) and now.minute == 2 and now.second < 30:

                print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] ⏰ Cycle de paiement de funding")

                regime = read_regime()
                best   = read_best_funding()

                # ── EN POSITION ───────────────────────────────
                if state['position'] == 'LONG_SPOT_SHORT_PERP':
                    sym = state['symbol']
                    spot_price, perp_price = get_prices(spot_ex, perp_ex, sym)
                    current_fr = get_current_funding_rate(perp_ex, sym)

                    # Funding collecté ce cycle
                    funding_usd = current_fr * TRADE_AMOUNT_USD
                    state['funding_payments'].append(current_fr * 100)
                    state['funding_collected_usd'] += funding_usd

                    upnl = compute_unrealized_pnl(state, spot_price, perp_price)
                    days_held = (datetime.now() - datetime.fromisoformat(state['entry_time'])).days

                    print(f"  📊 {sym} | Spot: ${spot_price:.2f} | Perp: ${perp_price:.2f}")
                    print(f"  💰 Funding ce cycle  : {current_fr*100:+.4f}% (${funding_usd:+.4f})")
                    print(f"  💰 Funding cumulé    : ${state['funding_collected_usd']:+.4f}")
                    print(f"  📈 PnL latent (basis): ${upnl:+.4f}")
                    print(f"  ⏱️  Durée             : {days_held}j")

                    # Log equity
                    log_equity(
                        equity=5000 + state['funding_collected_usd'] + upnl,
                        position=state['position'],
                        symbol=sym,
                        funding_collected=state['funding_collected_usd'],
                        unrealized_pnl=upnl,
                    )

                    # Vérification sortie
                    exit_flag, exit_reason = should_exit(state, current_fr)
                    if exit_flag:
                        print(f"\n  🚪 SORTIE : {exit_reason}")
                        exit_position(spot_ex, perp_ex, state)
                        log_trade("EXIT", sym, spot_price, perp_price, current_fr,
                                  exit_reason=exit_reason,
                                  total_funding=state['funding_collected_usd'])
                        state = load_state()  # reset
                        state['position'] = 'FLAT'
                        save_state(state)
                        print(f"  ✅ Position fermée. Funding total collecté : ${state.get('funding_collected_usd', 0):.4f}")

                # ── PAS EN POSITION ───────────────────────────
                elif state['position'] == 'FLAT':
                    # Log équité même en FLAT pour que le dashboard ne soit pas vide
                    log_equity(
                        equity=5000 + state.get('funding_collected_usd', 0.0),
                        position='FLAT',
                        symbol=None,
                        funding_collected=state.get('funding_collected_usd', 0.0),
                        unrealized_pnl=0.0,
                    )

                    # Vérification régime
                    if regime:
                        r = regime['regime']
                        if regime['strategies'].get('funding_carry') != True:
                            print(f"  🚫 Funding carry désactivé (régime {r})")
                            log_trade("REJECTED_REGIME", best['symbol'] if best else None,
                                      0, 0, 0, exit_reason=f"regime={r}")
                            time.sleep(60)
                            continue

                    # Vérification paire
                    if not best or not best.get('valid'):
                        print("  🚫 Aucune opportunité valide — APR insuffisant")
                        log_trade("REJECTED_NO_PAIR", best['symbol'] if best else None,
                                  0, 0, 0, exit_reason="no_valid_opportunity")
                        time.sleep(60)
                        continue

                    sym = best['symbol']
                    spot_price, perp_price = get_prices(spot_ex, perp_ex, sym)
                    current_fr = get_current_funding_rate(perp_ex, sym)
                    current_apr = current_fr * 3 * 365 * 100

                    print(f"  📊 Opportunité : {sym}")
                    print(f"     APR actuel  : {current_apr:.2f}%")
                    print(f"     APR 30j     : {best['apr_30d']}%")
                    print(f"     Breakeven   : {best['breakeven_days']} jours")

                    # Entrée seulement si le funding actuel est positif
                    if current_fr <= 0:
                        print(f"  🚫 Funding actuel négatif ({current_fr*100:.4f}%) — on attend")
                        log_trade("REJECTED_NEG_FUNDING", sym, spot_price, perp_price,
                                  current_fr, exit_reason="funding_rate<=0")
                        time.sleep(60)
                        continue

                    print(f"\n  🚀 ENTRÉE Long Spot + Short Perp sur {sym}")
                    result = enter_position(spot_ex, perp_ex, sym, spot_price, perp_price)

                    if result:
                        spot_size, perp_size = result
                        log_trade("ENTRY", sym, spot_price, perp_price, current_fr)
                        state = {
                            'position'          : 'LONG_SPOT_SHORT_PERP',
                            'symbol'            : sym,
                            'entry_time'        : datetime.now().isoformat(),
                            'entry_spot_price'  : spot_price,
                            'entry_perp_price'  : perp_price,
                            'spot_size'         : spot_size,
                            'perp_size'         : perp_size,
                            'funding_payments'  : [current_fr * 100],
                            'funding_collected_usd': current_fr * TRADE_AMOUNT_USD,
                        }
                        save_state(state)
                        print(f"  ✅ Position ouverte. Prochain check dans 8h.")

                time.sleep(30)
            else:
                time.sleep(20)

        except Exception as e:
            print(f"❌ Erreur : {e}")
            time.sleep(30)


if __name__ == "__main__":
    main()