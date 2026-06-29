import ccxt
import pandas as pd
import numpy as np
import joblib
import time
import csv
import os
import json
import sys
import requests as _requests
from datetime import datetime
from pathlib import Path

# ── Accès à config.py (à la racine du projet) ─────────────────
ROOT_DIR = Path(__file__).parent.parent.parent  # ML_Crypto_Bot/
sys.path.insert(0, str(ROOT_DIR))
from config import PATHS

# --- Chemins (centralisés dans config.py) ---
DATA_DIR = PATHS['statarb_data']
DATA_DIR.mkdir(exist_ok=True)

MODEL_FILE  = PATHS['statarb_model']
EQUITY_CSV  = PATHS['live_equity']
TRADES_CSV  = PATHS['live_trades']
REGIME_JSON = PATHS['regime_json']
PAIR_JSON   = PATHS['best_pair_json']

from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

# --- Paramètres de Trading ---
TRADE_AMOUNT_USD = 50.0
THRESHOLD_ML     = 0.60
WINDOW           = 200
SL_ZSCORE        = 6.0    # Stop-Loss : fermeture forcée si Z s'écarte trop
SL_PNL           = -0.05  # Stop-Loss : fermeture forcée si PnL latent < -5%
TIME_LIMIT       = 192    # Stop-Loss temporel : 48h max (192 bougies de 15m)

# --- Paire par défaut (si select_pair.py n'a pas encore tourné) ---
DEFAULT_SYM1 = 'AAVE/USDT'
DEFAULT_SYM2 = 'ETH/USDT'

# Variables globales — mises à jour dans main() selon le régime
SYM1 = DEFAULT_SYM1
SYM2 = DEFAULT_SYM2

# ── Lecture régime et paire ────────────────────────────────────
def read_regime():
    """Lit le régime actuel depuis Regime_Detector/data/current_regime.json"""
    try:
        with open(REGIME_JSON) as f:
            data = json.load(f)
        ts  = datetime.strptime(data['timestamp'], '%Y-%m-%d %H:%M:%S')
        age = (datetime.now() - ts).total_seconds() / 3600
        if age > 2:
            print(f"  ⚠️  Régime obsolète ({age:.1f}h) — lance live_regime.py")
        return data
    except FileNotFoundError:
        print("  ⚠️  Régime non disponible — lance compute_regime.py d'abord")
        return None

def read_best_pair():
    """Lit la meilleure paire depuis Regime_Detector/data/best_pair.json"""
    try:
        with open(PAIR_JSON) as f:
            data = json.load(f)
        ts  = datetime.strptime(data['updated'], '%Y-%m-%d %H:%M:%S')
        age = (datetime.now() - ts).total_seconds() / 3600 / 24
        if age > 8:
            print(f"  ⚠️  Scan paire obsolète ({age:.1f}j) — relance select_pair.py")
        return data
    except FileNotFoundError:
        print(f"  ⚠️  Aucun scan paire — utilisation paire par défaut ({DEFAULT_SYM1}/{DEFAULT_SYM2})")
        return None

def init_csv():
    """Crée les fichiers CSV s'ils n'existent pas avec leurs en-têtes"""
    if not os.path.exists(EQUITY_CSV):
        with open(EQUITY_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'equity_usdt', 'position_status',
                'unrealized_pnl_usdt', 'unrealized_pnl_pct', 'num_trades_total'
            ])

    if not os.path.exists(TRADES_CSV):
        with open(TRADES_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'action', 'zscore', 'ml_prob',
                'aave_price', 'eth_price', 'hedge_ratio', 'spread_value',
                'pnl_pct', 'duration_candles', 'exit_reason'
            ])

def log_equity(exchange, position, unrealized_pnl_usdt, unrealized_pnl_pct, num_trades_total):
    """Enregistre le solde actuel avec métriques enrichies"""
    try:
        balance = exchange.fetch_balance()
        usdt_total = balance['total'].get('USDT', 0)
        with open(EQUITY_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                round(usdt_total, 4),
                position,
                round(unrealized_pnl_usdt, 4),
                round(unrealized_pnl_pct, 4),
                num_trades_total
            ])
        supabase_insert('live_equity', {
            'timestamp'           : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'equity_usdt'         : round(usdt_total, 4),
            'position_status'     : position,
            'unrealized_pnl_usdt' : round(unrealized_pnl_usdt, 4),
            'unrealized_pnl_pct'  : round(unrealized_pnl_pct, 4),
            'num_trades_total'    : num_trades_total,
        })
        return usdt_total
    except Exception as e:
        print(f"Erreur de lecture du solde: {e}")
        return None

def log_trade(action, zscore, ml_prob, aave_price, eth_price,
              hedge_ratio=None, spread_value=None,
              pnl_pct=None, duration_candles=None, exit_reason=None):
    """Enregistre une action de trading avec métriques enrichies"""
    with open(TRADES_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            action,
            round(zscore, 4),
            round(ml_prob, 4),
            aave_price,
            eth_price,
            round(hedge_ratio, 6) if hedge_ratio is not None else '',
            round(spread_value, 6) if spread_value is not None else '',
            round(pnl_pct * 100, 4) if pnl_pct is not None else '',
            duration_candles if duration_candles is not None else '',
            exit_reason if exit_reason is not None else ''
        ])
    supabase_insert('live_trades', {
        'timestamp'       : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'action'          : action,
        'zscore'          : round(zscore, 4),
        'ml_prob'         : round(ml_prob, 4),
        'aave_price'      : aave_price,
        'eth_price'       : eth_price,
        'hedge_ratio'     : round(hedge_ratio, 6)   if hedge_ratio      is not None else None,
        'spread_value'    : round(spread_value, 6)  if spread_value     is not None else None,
        'pnl_pct'         : round(pnl_pct, 6)       if pnl_pct          is not None else None,
        'duration_candles': duration_candles,
        'exit_reason'     : exit_reason,
        'sym1'            : SYM1,
        'sym2'            : SYM2,
    })

def init_exchange():
    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
        'options': {'defaultType': 'future', 'adjustForTimeDifference': True}
    })
    exchange.enable_demo_trading(True)
    return exchange

# ── Supabase logging ───────────────────────────────────────────
_SB_URL = os.getenv('SUPABASE_URL', '')
_SB_KEY = os.getenv('SUPABASE_KEY', '')

def supabase_insert(table, data):
    """Écrit une ligne dans Supabase — silencieux si erreur"""
    if not _SB_URL or not _SB_KEY:
        return
    try:
        _requests.post(
            f"{_SB_URL}/rest/v1/{table}",
            headers={
                'apikey'       : _SB_KEY,
                'Authorization': f'Bearer {_SB_KEY}',
                'Content-Type' : 'application/json',
                'Prefer'       : 'return=minimal',
            },
            json=data,
            timeout=3,
        )
    except:
        pass  # Ne jamais bloquer le bot pour un log

def fetch_latest_data(exchange):
    ohlcv1 = exchange.fetch_ohlcv(SYM1, '15m', limit=500)
    ohlcv2 = exchange.fetch_ohlcv(SYM2, '15m', limit=500)

    df1 = pd.DataFrame(ohlcv1, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df2 = pd.DataFrame(ohlcv2, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    df1['timestamp'] = pd.to_datetime(df1['timestamp'], unit='ms')
    df1.set_index('timestamp', inplace=True)
    df2['timestamp'] = pd.to_datetime(df2['timestamp'], unit='ms')
    df2.set_index('timestamp', inplace=True)

    df = pd.concat([df1['close'], df2['close']], axis=1).dropna()
    df.columns = ['AAVE', 'ETH']
    return df

def get_current_features(df):
    rolling_cov = df['AAVE'].rolling(WINDOW).cov(df['ETH'])
    rolling_var = df['ETH'].rolling(WINDOW).var()
    df['ratio'] = rolling_cov / rolling_var
    df['spread'] = df['AAVE'] - (df['ratio'] * df['ETH'])
    df['zscore'] = (df['spread'] - df['spread'].rolling(WINDOW).mean()) / df['spread'].rolling(WINDOW).std()

    df['zscore_mom_1h'] = df['zscore'].diff(4)
    df['zscore_mom_4h'] = df['zscore'].diff(16)
    df['aave_ret_1h'] = df['AAVE'].pct_change(4)
    df['eth_ret_1h'] = df['ETH'].pct_change(4)
    df['aave_vol_4h'] = df['AAVE'].pct_change().rolling(16).std()
    df['eth_vol_4h'] = df['ETH'].pct_change().rolling(16).std()

    return df.iloc[-1]

def compute_unrealized_pnl(position, direction,
                            entry_price_aave, entry_price_eth,
                            current_aave, current_eth):
    """Calcule le PnL latent de la position ouverte"""
    if position == "FLAT":
        return 0.0, 0.0

    ret_aave = (current_aave - entry_price_aave) / entry_price_aave
    ret_eth  = (entry_price_eth - current_eth)   / entry_price_eth  # Short ETH

    if direction == -1:  # SHORT_SPREAD : on inverse
        ret_aave = -ret_aave
        ret_eth  = -ret_eth

    gross_pnl_pct = (ret_aave + ret_eth) / 2
    pnl_usdt = TRADE_AMOUNT_USD * gross_pnl_pct
    return pnl_usdt, gross_pnl_pct

def execute_trade(exchange, direction, current_prices, zscore, ml_prob,
                  pos_aave_size=0, pos_eth_size=0,
                  entry_price_aave=0, entry_price_eth=0,
                  entry_candle=0, current_candle=0,
                  hedge_ratio=None, spread_value=None, exit_reason=None):

    aave_price = current_prices['AAVE']
    eth_price  = current_prices['ETH']

    aave_size = round(TRADE_AMOUNT_USD / aave_price, 1)
    eth_size  = round(TRADE_AMOUNT_USD / eth_price, 3)

    print(f"💸 Tentative d'exécution : {direction}")
    print(f"   Tailles calculées -> AAVE: {aave_size} | ETH: {eth_size}")

    if direction == "ENTRY_LONG_SPREAD":
        try:
            exchange.create_market_buy_order(SYM1, aave_size)
        except Exception as e:
            print(f"❌ Échec ordre AAVE (buy) : {e}. Trade annulé.")
            return "FLAT", 0, 0, 0, 0
        try:
            exchange.create_market_sell_order(SYM2, eth_size)
        except Exception as e:
            print(f"❌ Échec ordre ETH (sell) : {e}. Annulation de la patte AAVE...")
            try:
                exchange.create_market_sell_order(SYM1, aave_size)
                print("↩️ Patte AAVE annulée avec succès.")
            except Exception as e2:
                print(f"🚨 ALERTE : Impossible d'annuler la patte AAVE : {e2}. Position partielle ouverte !")
            return "FLAT", 0, 0, 0, 0
        log_trade("LONG_SPREAD", zscore, ml_prob, aave_price, eth_price,
                  hedge_ratio=hedge_ratio, spread_value=spread_value)
        print("✅ Spread Ouvert (Long AAVE / Short ETH)")
        return "LONG_SPREAD", aave_size, eth_size, aave_price, eth_price

    elif direction == "ENTRY_SHORT_SPREAD":
        try:
            exchange.create_market_sell_order(SYM1, aave_size)
        except Exception as e:
            print(f"❌ Échec ordre AAVE (sell) : {e}. Trade annulé.")
            return "FLAT", 0, 0, 0, 0
        try:
            exchange.create_market_buy_order(SYM2, eth_size)
        except Exception as e:
            print(f"❌ Échec ordre ETH (buy) : {e}. Annulation de la patte AAVE...")
            try:
                exchange.create_market_buy_order(SYM1, aave_size)
                print("↩️ Patte AAVE annulée avec succès.")
            except Exception as e2:
                print(f"🚨 ALERTE : Impossible d'annuler la patte AAVE : {e2}. Position partielle ouverte !")
            return "FLAT", 0, 0, 0, 0
        log_trade("SHORT_SPREAD", zscore, ml_prob, aave_price, eth_price,
                  hedge_ratio=hedge_ratio, spread_value=spread_value)
        print("✅ Spread Ouvert (Short AAVE / Long ETH)")
        return "SHORT_SPREAD", aave_size, eth_size, aave_price, eth_price

    elif direction in ("EXIT_LONG_SPREAD", "EXIT_SHORT_SPREAD"):
        # Calcul PnL réalisé
        if direction == "EXIT_LONG_SPREAD":
            ret_aave = (aave_price - entry_price_aave) / entry_price_aave
            ret_eth  = (entry_price_eth - eth_price)   / entry_price_eth
        else:
            ret_aave = (entry_price_aave - aave_price) / entry_price_aave
            ret_eth  = (eth_price - entry_price_eth)   / entry_price_eth
        pnl_pct = (ret_aave + ret_eth) / 2
        duration = current_candle - entry_candle

        # Exécution des ordres de sortie
        sl1 = SYM1
        if direction == "EXIT_LONG_SPREAD":
            try:
                exchange.create_market_sell_order(SYM1, pos_aave_size)
            except Exception as e:
                print(f"❌ Échec fermeture AAVE (sell) : {e}. Retry dans 10s...")
                time.sleep(10)
                try:
                    exchange.create_market_sell_order(SYM1, pos_aave_size)
                except Exception as e2:
                    print(f"🚨 ALERTE : Impossible de fermer AAVE : {e2}. Intervention manuelle requise !")
            try:
                exchange.create_market_buy_order(SYM2, pos_eth_size)
            except Exception as e:
                print(f"❌ Échec fermeture ETH (buy) : {e}. Retry dans 10s...")
                time.sleep(10)
                try:
                    exchange.create_market_buy_order(SYM2, pos_eth_size)
                except Exception as e2:
                    print(f"🚨 ALERTE : Impossible de fermer ETH : {e2}. Intervention manuelle requise !")
        else:
            try:
                exchange.create_market_buy_order(SYM1, pos_aave_size)
            except Exception as e:
                print(f"❌ Échec fermeture AAVE (buy) : {e}. Retry dans 10s...")
                time.sleep(10)
                try:
                    exchange.create_market_buy_order(SYM1, pos_aave_size)
                except Exception as e2:
                    print(f"🚨 ALERTE : Impossible de fermer AAVE : {e2}. Intervention manuelle requise !")
            try:
                exchange.create_market_sell_order(SYM2, pos_eth_size)
            except Exception as e:
                print(f"❌ Échec fermeture ETH (sell) : {e}. Retry dans 10s...")
                time.sleep(10)
                try:
                    exchange.create_market_sell_order(SYM2, pos_eth_size)
                except Exception as e2:
                    print(f"🚨 ALERTE : Impossible de fermer ETH : {e2}. Intervention manuelle requise !")

        log_trade("EXIT", zscore, ml_prob, aave_price, eth_price,
                  hedge_ratio=hedge_ratio, spread_value=spread_value,
                  pnl_pct=pnl_pct, duration_candles=duration, exit_reason=exit_reason)
        print(f"✅ Spread Fermé. PnL : {pnl_pct*100:+.3f}% | Durée : {duration} bougies ({duration*15}min)")
        return "FLAT", 0, 0, 0, 0


def recover_position(exchange):
    """
    Au démarrage, interroge Binance pour détecter une position déjà ouverte.
    Retourne l'état reconstruit : position, direction, tailles, prix d'entrée.
    """
    print("🔍 Vérification des positions ouvertes sur Binance...")
    try:
        positions = exchange.fetch_positions([SYM1, SYM2])
        print(f"   Raw positions reçues : {len(positions)}")
        for p in positions:
            print(f"   → symbol={p.get('symbol')} | side={p.get('side')} | contracts={p.get('contracts')} | entryPrice={p.get('entryPrice')}")
        open_pos = {
            p['symbol']: p for p in positions
            if p.get('contracts') is not None and float(p.get('contracts') or 0) != 0
        }

        # Binance retourne 'AAVE/USDT:USDT' — on cherche par correspondance partielle
        aave_key = next((k for k in open_pos if 'AAVE' in k), None)
        eth_key  = next((k for k in open_pos if 'ETH'  in k and 'AAVE' not in k), None)

        has_aave = aave_key is not None
        has_eth  = eth_key  is not None

        if not has_aave and not has_eth:
            print("✅ Aucune position ouverte — démarrage propre.")
            return "FLAT", 0, 0, 0, 0, 0

        if has_aave and has_eth:
            aave_pos = open_pos[aave_key]
            eth_pos  = open_pos[eth_key]

            aave_size        = abs(float(aave_pos['contracts']))
            eth_size         = abs(float(eth_pos['contracts']))
            entry_price_aave = float(aave_pos['entryPrice'])
            entry_price_eth  = float(eth_pos['entryPrice'])

            # Long AAVE + Short ETH = LONG_SPREAD
            # Short AAVE + Long ETH = SHORT_SPREAD
            if aave_pos['side'] == 'long' and eth_pos['side'] == 'short':
                position      = "LONG_SPREAD"
                direction_int = 1
            elif aave_pos['side'] == 'short' and eth_pos['side'] == 'long':
                position      = "SHORT_SPREAD"
                direction_int = -1
            else:
                print(f"⚠️ Configuration inattendue : AAVE={aave_pos['side']} / ETH={eth_pos['side']}")
                print("   Intervention manuelle recommandée.")
                return "FLAT", 0, 0, 0, 0, 0

            print(f"♻️  Position récupérée : {position}")
            print(f"   AAVE : {aave_pos['side']} {aave_size} @ {entry_price_aave}")
            print(f"   ETH  : {eth_pos['side']}  {eth_size} @ {entry_price_eth}")
            return position, direction_int, aave_size, eth_size, entry_price_aave, entry_price_eth

        else:
            print("🚨 ALERTE : Une seule patte détectée — position non couverte !")
            print("   Intervention manuelle recommandée.")
            return "FLAT", 0, 0, 0, 0, 0

    except Exception as e:
        print(f"⚠️ Impossible de récupérer les positions : {e}")
        print("   Démarrage en mode FLAT par sécurité.")
        return "FLAT", 0, 0, 0, 0, 0


def main():
    print("="*60)
    print("🟢 STAT-ARB BOT EN LIGNE (BINANCE DEMO)")
    print("="*60)

    init_csv()
    exchange = init_exchange()
    model = joblib.load(MODEL_FILE)

    initial_equity = log_equity(exchange, "FLAT", 0.0, 0.0, 0)
    print(f"✅ API connectée. Capital de départ : {initial_equity:.2f} USDT")

    # ── Lecture régime et meilleure paire ─────────────────────
    global SYM1, SYM2
    regime    = read_regime()
    best_pair = read_best_pair()

    # Sélection de la paire (best_pair si valide, sinon défaut)
    if best_pair and best_pair.get('valid'):
        SYM1 = best_pair['sym1']
        SYM2 = best_pair['sym2']
        print(f"📊 Paire sélectionnée : {SYM1} / {SYM2} (score={best_pair['score']})")
    else:
        SYM1, SYM2 = DEFAULT_SYM1, DEFAULT_SYM2
        print(f"📊 Paire par défaut   : {SYM1} / {SYM2}")

    # Vérification régime initial
    if regime:
        emoji = {'MEAN_REV':'🟢','TRENDING':'🟡','HIGH_VOL':'🔴','NEUTRAL':'⚪'}
        r = regime['regime']
        print(f"🎯 Régime actuel      : {emoji.get(r,'⚪')} {r} "
              f"({'stat arb ✅' if regime['strategies']['stat_arb'] else 'stat arb ❌'})")

    # Récupération automatique de la position si redémarrage en cours de trade
    position, direction_int, pos_aave_size, pos_eth_size, entry_price_aave, entry_price_eth = recover_position(exchange)

    entry_candle     = 0
    candle_count     = 0
    num_trades_total = 0
    last_regime_check = datetime.now()  # Pour refresh horaire

    while True:
        try:
            now = datetime.now()
            if now.minute % 15 == 0 and now.second < 10:
                print(f"\n[{now.strftime('%H:%M:%S')}] 🔄 Scan du marché...")
                candle_count += 1

                # Refresh régime toutes les heures
                if (datetime.now() - last_regime_check).total_seconds() >= 3600:
                    regime = read_regime()
                    best_pair = read_best_pair()
                    if best_pair and best_pair.get('valid'):
                        SYM1 = best_pair['sym1']
                        SYM2 = best_pair['sym2']
                    last_regime_check = datetime.now()
                    if regime:
                        emoji = {'MEAN_REV':'🟢','TRENDING':'🟡','HIGH_VOL':'🔴','NEUTRAL':'⚪'}
                        r = regime['regime']
                        print(f"  🎯 Régime : {emoji.get(r,'⚪')} {r}")

                df = fetch_latest_data(exchange)
                current_state = get_current_features(df)
                z            = current_state['zscore']
                hedge_ratio  = current_state['ratio']
                spread_value = current_state['spread']
                aave_now     = current_state['AAVE']
                eth_now      = current_state['ETH']

                # Calcul PnL latent
                upnl_usdt, upnl_pct = compute_unrealized_pnl(
                    position, direction_int,
                    entry_price_aave, entry_price_eth,
                    aave_now, eth_now
                )

                current_equity = log_equity(exchange, position, upnl_usdt, upnl_pct, num_trades_total)

                if np.isnan(z):
                    print("⚠️ Z-Score NaN — pas assez de données, on skip cette bougie.")
                    time.sleep(60)
                    continue

                print(f"📊 Z-Score : {z:.2f} | Capital : {current_equity:.2f} USDT | PnL latent : {upnl_pct*100:+.3f}%")

                # --- LOGIQUE DE SORTIE ---
                candles_in_trade = candle_count - entry_candle

                if position == "LONG_SPREAD":
                    if candles_in_trade >= TIME_LIMIT:
                        print(f"⏰ BARRIÈRE TEMPORELLE ({candles_in_trade} bougies = {candles_in_trade*15//60}h). Fermeture forcée.")
                        position, pos_aave_size, pos_eth_size, entry_price_aave, entry_price_eth = execute_trade(
                            exchange, "EXIT_LONG_SPREAD",
                            {'AAVE': aave_now, 'ETH': eth_now},
                            z, 0, pos_aave_size, pos_eth_size,
                            entry_price_aave, entry_price_eth,
                            entry_candle, candle_count,
                            hedge_ratio, spread_value, exit_reason="TIME_LIMIT"
                        )
                        direction_int = 0
                    elif z <= -SL_ZSCORE or upnl_pct <= SL_PNL:
                        reason = "STOP_LOSS_Z" if z <= -SL_ZSCORE else "STOP_LOSS_PNL"
                        print(f"🛑 STOP-LOSS DÉCLENCHÉ ({reason} | Z={z:.2f} | PnL={upnl_pct*100:.2f}%). Fermeture forcée.")
                        position, pos_aave_size, pos_eth_size, entry_price_aave, entry_price_eth = execute_trade(
                            exchange, "EXIT_LONG_SPREAD",
                            {'AAVE': aave_now, 'ETH': eth_now},
                            z, 0, pos_aave_size, pos_eth_size,
                            entry_price_aave, entry_price_eth,
                            entry_candle, candle_count,
                            hedge_ratio, spread_value, exit_reason=reason
                        )
                        direction_int = 0
                    elif z >= 0:
                        print("🎯 Take Profit (Z >= 0). Fermeture du Long Spread.")
                        position, pos_aave_size, pos_eth_size, entry_price_aave, entry_price_eth = execute_trade(
                            exchange, "EXIT_LONG_SPREAD",
                            {'AAVE': aave_now, 'ETH': eth_now},
                            z, 0, pos_aave_size, pos_eth_size,
                            entry_price_aave, entry_price_eth,
                            entry_candle, candle_count,
                            hedge_ratio, spread_value, exit_reason="TP"
                        )
                        direction_int = 0

                elif position == "SHORT_SPREAD":
                    if candles_in_trade >= TIME_LIMIT:
                        print(f"⏰ BARRIÈRE TEMPORELLE ({candles_in_trade} bougies = {candles_in_trade*15//60}h). Fermeture forcée.")
                        position, pos_aave_size, pos_eth_size, entry_price_aave, entry_price_eth = execute_trade(
                            exchange, "EXIT_SHORT_SPREAD",
                            {'AAVE': aave_now, 'ETH': eth_now},
                            z, 0, pos_aave_size, pos_eth_size,
                            entry_price_aave, entry_price_eth,
                            entry_candle, candle_count,
                            hedge_ratio, spread_value, exit_reason="TIME_LIMIT"
                        )
                        direction_int = 0
                    elif z >= SL_ZSCORE or upnl_pct <= SL_PNL:
                        reason = "STOP_LOSS_Z" if z >= SL_ZSCORE else "STOP_LOSS_PNL"
                        print(f"🛑 STOP-LOSS DÉCLENCHÉ ({reason} | Z={z:.2f} | PnL={upnl_pct*100:.2f}%). Fermeture forcée.")
                        position, pos_aave_size, pos_eth_size, entry_price_aave, entry_price_eth = execute_trade(
                            exchange, "EXIT_SHORT_SPREAD",
                            {'AAVE': aave_now, 'ETH': eth_now},
                            z, 0, pos_aave_size, pos_eth_size,
                            entry_price_aave, entry_price_eth,
                            entry_candle, candle_count,
                            hedge_ratio, spread_value, exit_reason=reason
                        )
                        direction_int = 0
                    elif z <= 0:
                        print("🎯 Take Profit (Z <= 0). Fermeture du Short Spread.")
                        position, pos_aave_size, pos_eth_size, entry_price_aave, entry_price_eth = execute_trade(
                            exchange, "EXIT_SHORT_SPREAD",
                            {'AAVE': aave_now, 'ETH': eth_now},
                            z, 0, pos_aave_size, pos_eth_size,
                            entry_price_aave, entry_price_eth,
                            entry_candle, candle_count,
                            hedge_ratio, spread_value, exit_reason="TP"
                        )
                        direction_int = 0

                # --- LOGIQUE D'ENTRÉE ---
                elif position == "FLAT":
                    # Vérification régime avant toute entrée
                    if regime and not regime['strategies']['stat_arb']:
                        r = regime['regime']
                        print(f"  🚫 Stat arb désactivé (régime {r}) — aucune entrée.")
                    elif z <= -2.0 or z >= 2.0:
                        print("⚠️ Anomalie détectée ! Consultation du ML...")

                        features = [
                            current_state['zscore'],       current_state['zscore_mom_1h'],
                            current_state['zscore_mom_4h'], current_state['aave_ret_1h'],
                            current_state['eth_ret_1h'],   current_state['aave_vol_4h'],
                            current_state['eth_vol_4h']
                        ]

                        prob = model.predict_proba([features])[0][1]
                        print(f"🧠 Confiance ML : {prob*100:.2f}%")

                        if prob > THRESHOLD_ML:
                            dir_str = "ENTRY_LONG_SPREAD" if z <= -2.0 else "ENTRY_SHORT_SPREAD"
                            direction_int = 1 if z <= -2.0 else -1
                            position, pos_aave_size, pos_eth_size, entry_price_aave, entry_price_eth = execute_trade(
                                exchange, dir_str,
                                {'AAVE': aave_now, 'ETH': eth_now},
                                z, prob,
                                hedge_ratio=hedge_ratio, spread_value=spread_value
                            )
                            if position != "FLAT":
                                entry_candle = candle_count
                                num_trades_total += 1
                        else:
                            print("🚫 Trade rejeté par le vigile ML.")
                            log_trade("REJECTED_BY_ML", z, prob, aave_now, eth_now,
                                      hedge_ratio=hedge_ratio, spread_value=spread_value)

                time.sleep(60)
            else:
                time.sleep(1)

        except Exception as e:
            print(f"❌ Erreur critique : {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()