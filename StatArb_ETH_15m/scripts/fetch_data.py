import ccxt
import pandas as pd
import time
from pathlib import Path
from datetime import datetime, timedelta

# --- Configuration des chemins (S'adapte à ta nouvelle structure) ---
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
FILE_PATH = DATA_DIR / "historical_AAVE_ETH_15m.csv"

def fetch_ohlcv_history(symbol, timeframe='15m', days=180):
    """
    Télécharge l'historique massif en contournant la limite des 1000 bougies de Binance.
    """
    exchange = ccxt.binance({'enableRateLimit': True})
    
    # Calcul du timestamp de départ en millisecondes
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    all_ohlcv = []
    
    print(f"📥 Aspiration de {symbol} sur les {days} derniers jours...")
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            
            # Le prochain 'since' est le timestamp de la dernière bougie téléchargée + 1ms
            since = ohlcv[-1][0] + 1
            
            # Si la dernière bougie téléchargée est proche de l'heure actuelle, on a fini
            if ohlcv[-1][0] >= int(datetime.now().timestamp() * 1000) - (15 * 60 * 1000):
                break
                
        except Exception as e:
            print(f"⚠️ Erreur de connexion API : {e}. Nouvelle tentative dans 2s...")
            time.sleep(2)
            
    # Conversion en DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Nettoyage des doublons potentiels (sécurité)
    df = df[~df.index.duplicated(keep='first')]
    return df['close']

def main():
    print("="*60)
    print("🚀 ÉTAPE 1 : DATA ENGINEERING (FETCH HISTORY)")
    print("="*60)
    
    # On vise 6 mois d'historique (180 jours)
    days_to_fetch = 180 
    
    # Téléchargement
    df_aave = fetch_ohlcv_history('AAVE/USDT', '15m', days_to_fetch)
    df_eth = fetch_ohlcv_history('ETH/USDT', '15m', days_to_fetch)
    
    # 2. Alignement strict des deux séries temporelles
    print("\n⚙️ Alignement et nettoyage des données...")
    df = pd.concat([df_aave, df_eth], axis=1).dropna()
    df.columns = ['AAVE', 'ETH']
    
    print(f"✅ Données prêtes : {len(df)} bougies (Timeframe: 15m).")
    print(f"📅 Période : du {df.index[0]} au {df.index[-1]}")
    
    # 3. Sauvegarde dans le dossier /data
    df.to_csv(FILE_PATH)
    print(f"💾 Sauvegardé avec succès dans : {FILE_PATH}")
    print("="*60)

if __name__ == "__main__":
    main()