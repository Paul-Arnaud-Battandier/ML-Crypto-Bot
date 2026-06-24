import ccxt
import pandas as pd
import numpy as np
import joblib
import time
import csv
import os
from datetime import datetime
from pathlib import Path

# --- Chemins et Clés ---
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

MODEL_FILE = ROOT_DIR / "model" / "lgbm_statarb.pkl"
EQUITY_CSV = DATA_DIR / "live_equity.csv"
TRADES_CSV = DATA_DIR / "live_trades.csv"

from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

# --- Paramètres de Trading ---
SYM1, SYM2 = 'AAVE/USDT', 'ETH/USDT'
TRADE_AMOUNT_USD = 50.0  
THRESHOLD_ML = 0.60      
WINDOW = 200             

def init_csv():
    """Crée les fichiers CSV s'ils n'existent pas avec leurs en-têtes"""
    if not os.path.exists(EQUITY_CSV):
        with open(EQUITY_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'equity_usdt'])
            
    if not os.path.exists(TRADES_CSV):
        with open(TRADES_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'action', 'zscore', 'ml_prob', 'aave_price', 'eth_price'])

def log_equity(exchange):
    """Enregistre le solde actuel"""
    try:
        balance = exchange.fetch_balance()
        usdt_total = balance['total'].get('USDT', 0)
        with open(EQUITY_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), usdt_total])
        return usdt_total
    except Exception as e:
        print(f"Erreur de lecture du solde: {e}")
        return None

def log_trade(action, zscore, ml_prob, aave_price, eth_price):
    """Enregistre une action de trading"""
    with open(TRADES_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), action, round(zscore, 2), round(ml_prob, 2), aave_price, eth_price])

def init_exchange():
    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
        'options': {'defaultType': 'future', 'adjustForTimeDifference': True}
    })
    exchange.enable_demo_trading(True)
    return exchange

def fetch_latest_data(exchange):
    ohlcv1 = exchange.fetch_ohlcv(SYM1, '15m', limit=250)
    ohlcv2 = exchange.fetch_ohlcv(SYM2, '15m', limit=250)
    
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
    # --- LE VRAI HEDGE RATIO INSTITUTIONNEL (Rolling OLS Vectorisé) ---
    # Covariance glissante entre AAVE et ETH
    rolling_cov = df['AAVE'].rolling(WINDOW).cov(df['ETH'])
    # Variance glissante de ETH
    rolling_var = df['ETH'].rolling(WINDOW).var()

    # Le Beta (La pente de la régression linéaire) devient notre Ratio dynamique
    df['ratio'] = rolling_cov / rolling_var

    # Le nouveau Spread parfaitement couvert
    df['spread'] = df['AAVE'] - (df['ratio'] * df['ETH'])
    df['zscore'] = (df['spread'] - df['spread'].rolling(WINDOW).mean()) / df['spread'].rolling(WINDOW).std()
    
    df['zscore_mom_1h'] = df['zscore'].diff(4)
    df['zscore_mom_4h'] = df['zscore'].diff(16)
    df['aave_ret_1h'] = df['AAVE'].pct_change(4)
    df['eth_ret_1h'] = df['ETH'].pct_change(4)
    df['aave_vol_4h'] = df['AAVE'].pct_change().rolling(16).std()
    df['eth_vol_4h'] = df['ETH'].pct_change().rolling(16).std()
    
    return df.iloc[-1]

def execute_trade(exchange, direction, zscore, ml_prob, aave_price=0.0, eth_price=0.0, aave_size=0.0, eth_size=0.0):
    """
    Exécute les ordres avec paramètres séparés et mécanisme de sécurité (Rollback) 
    en cas d'exécution partielle.
    """
    print(f"💸 Tentative d'exécution : {direction}")
    
    if direction == "ENTRY_LONG_SPREAD":
        a_size = round(TRADE_AMOUNT_USD / aave_price, 1)
        e_size = round(TRADE_AMOUNT_USD / eth_price, 3)
        print(f"   Tailles calculées -> AAVE: {a_size} | ETH: {e_size}")
        
        # 1. Tentative Patte AAVE
        try:
            exchange.create_market_buy_order(SYM1, a_size)
        except Exception as e:
            print(f"❌ Échec Patte 1 (AAVE) : {e}. Abandon de l'arbitrage.")
            return "FLAT", 0, 0
            
        # 2. Tentative Patte ETH
        try:
            exchange.create_market_sell_order(SYM2, e_size)
        except Exception as e:
            print(f"🚨 ERREUR CRITIQUE Patte 2 (ETH) : {e}.")
            print("   -> Lancement de la procédure de ROLLBACK (Revente de AAVE)...")
            exchange.create_market_sell_order(SYM1, a_size)
            return "FLAT", 0, 0
            
        log_trade("LONG_SPREAD", zscore, ml_prob, aave_price, eth_price)
        print("✅ Spread Ouvert (Long AAVE / Short ETH)")
        return "LONG_SPREAD", a_size, e_size

    elif direction == "ENTRY_SHORT_SPREAD":
        a_size = round(TRADE_AMOUNT_USD / aave_price, 1)
        e_size = round(TRADE_AMOUNT_USD / eth_price, 3)
        print(f"   Tailles calculées -> AAVE: {a_size} | ETH: {e_size}")
        
        try:
            exchange.create_market_sell_order(SYM1, a_size)
        except Exception as e:
            print(f"❌ Échec Patte 1 (AAVE) : {e}. Abandon de l'arbitrage.")
            return "FLAT", 0, 0
            
        try:
            exchange.create_market_buy_order(SYM2, e_size)
        except Exception as e:
            print(f"🚨 ERREUR CRITIQUE Patte 2 (ETH) : {e}.")
            print("   -> Lancement de la procédure de ROLLBACK (Rachat de AAVE)...")
            exchange.create_market_buy_order(SYM1, a_size)
            return "FLAT", 0, 0
            
        log_trade("SHORT_SPREAD", zscore, ml_prob, aave_price, eth_price)
        print("✅ Spread Ouvert (Short AAVE / Long ETH)")
        return "SHORT_SPREAD", a_size, e_size

    elif direction == "EXIT_LONG_SPREAD":
        try:
            exchange.create_market_sell_order(SYM1, aave_size)
            exchange.create_market_buy_order(SYM2, eth_size)
            print("✅ Spread Fermé. PnL Encaissé.")
        except Exception as e:
            print(f"🚨 ERREUR DE SORTIE (Intervention manuelle requise) : {e}")
            
        log_trade("EXIT", zscore, ml_prob, aave_price, eth_price)
        return "FLAT", 0, 0

    elif direction == "EXIT_SHORT_SPREAD":
        try:
            exchange.create_market_buy_order(SYM1, aave_size)
            exchange.create_market_sell_order(SYM2, eth_size)
            print("✅ Spread Fermé. PnL Encaissé.")
        except Exception as e:
            print(f"🚨 ERREUR DE SORTIE (Intervention manuelle requise) : {e}")
            
        log_trade("EXIT", zscore, ml_prob, aave_price, eth_price)
        return "FLAT", 0, 0

def main():
    print("="*60)
    print("🟢 STAT-ARB BOT EN LIGNE (BINANCE DEMO)")
    print("="*60)
    
    init_csv()
    exchange = init_exchange()
    model = joblib.load(MODEL_FILE)
    
    # Log initial du capital
    initial_equity = log_equity(exchange)
    print(f"✅ API connectée. Capital de départ : {initial_equity:.2f} USDT")
    
    position = "FLAT" 
    pos_aave_size = 0
    pos_eth_size = 0
    
    while True:
        try:
            now = datetime.now()
            if now.minute % 15 == 0 and now.second < 10:
                print(f"\n[{now.strftime('%H:%M:%S')}] 🔄 Scan du marché...")
                
                # Mise à jour du graphique de capital
                current_equity = log_equity(exchange)
                
                df = fetch_latest_data(exchange)
                current_state = get_current_features(df)
                z = current_state['zscore']
                
                print(f"📊 Z-Score actuel : {z:.2f} | Capital : {current_equity:.2f} USDT")
                
                # --- LOGIQUE DE SORTIE ---
                if position == "LONG_SPREAD" and z >= 0:
                    print("🎯 Objectif atteint (Z >= 0). Fermeture du Long Spread.")
                    position, pos_aave_size, pos_eth_size = execute_trade(
                        exchange, "EXIT_LONG_SPREAD", z, 0, 
                        aave_price=current_state['AAVE'], eth_price=current_state['ETH'], 
                        aave_size=pos_aave_size, eth_size=pos_eth_size
                    )
                    
                elif position == "SHORT_SPREAD" and z <= 0:
                    print("🎯 Objectif atteint (Z <= 0). Fermeture du Short Spread.")
                    position, pos_aave_size, pos_eth_size = execute_trade(
                        exchange, "EXIT_SHORT_SPREAD", z, 0, 
                        aave_price=current_state['AAVE'], eth_price=current_state['ETH'], 
                        aave_size=pos_aave_size, eth_size=pos_eth_size
                    )
                
                # --- LOGIQUE D'ENTRÉE ---
                elif position == "FLAT":
                    if z <= -2.0 or z >= 2.0:
                        print("⚠️ Anomalie détectée ! Consultation du ML...")
                        
                        features = [
                            current_state['zscore'], current_state['zscore_mom_1h'], 
                            current_state['zscore_mom_4h'], current_state['aave_ret_1h'], 
                            current_state['eth_ret_1h'], current_state['aave_vol_4h'], 
                            current_state['eth_vol_4h']
                        ]
                        
                        prob = model.predict_proba([features])[0][1]
                        print(f"🧠 Confiance ML : {prob*100:.2f}%")
                        
                        if prob > THRESHOLD_ML:
                            direction = "ENTRY_LONG_SPREAD" if z <= -2.0 else "ENTRY_SHORT_SPREAD"
                            position, pos_aave_size, pos_eth_size = execute_trade(
                                exchange, direction, z, prob, 
                                aave_price=current_state['AAVE'], eth_price=current_state['ETH']
                            )
                        else:
                            print("🚫 Trade rejeté par le vigile ML.")
                            # On log quand même le rejet pour les stats du site
                            log_trade("REJECTED_BY_ML", z, prob, current_state['AAVE'], current_state['ETH'])
                
                time.sleep(60)
            else:
                time.sleep(1)
                
        except Exception as e:
            print(f"❌ Erreur critique : {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()