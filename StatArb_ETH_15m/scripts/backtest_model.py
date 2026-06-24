import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# --- Chemins ---
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "model"

INPUT_FILE = DATA_DIR / "historical_AAVE_ETH_15m.csv"
MODEL_FILE = MODEL_DIR / "lgbm_statarb.pkl"

def main():
    print("="*70)
    print("🛡️ ÉTAPE 4 : BACKTEST OUT-OF-SAMPLE (RIGUEUR LOPEZ DE PRADO)")
    print("="*70)

    # 1. Chargement des données brutes
    df = pd.read_csv(INPUT_FILE, index_col='timestamp', parse_dates=True)
    
    # 2. Recalcul des Features (Exactement comme dans l'entraînement)
    print("⚙️ Recalcul des indicateurs...")
    window = 200
    df['ratio'] = df['AAVE'].rolling(window).mean() / df['ETH'].rolling(window).mean()
    df['spread'] = df['AAVE'] - (df['ratio'] * df['ETH'])
    df['zscore_entry'] = (df['spread'] - df['spread'].rolling(window).mean()) / df['spread'].rolling(window).std()
    
    df['zscore_mom_1h'] = df['zscore_entry'].diff(4)
    df['zscore_mom_4h'] = df['zscore_entry'].diff(16)
    df['aave_ret_1h'] = df['AAVE'].pct_change(4)
    df['eth_ret_1h'] = df['ETH'].pct_change(4)
    df['aave_vol_4h'] = df['AAVE'].pct_change().rolling(16).std()
    df['eth_vol_4h'] = df['ETH'].pct_change().rolling(16).std()
    
    df = df.dropna().copy()

    # 3. Isolation de l'Out-Of-Sample (Les 20% que le ML ne connaît pas)
    split_idx = int(len(df) * 0.8)
    df_test = df.iloc[split_idx:].copy()
    print(f"📅 Période de Backtest : du {df_test.index[0]} au {df_test.index[-1]}")
    print(f"📊 Nombre de bougies simulées : {len(df_test)}")

    # 4. Prédiction des probabilités ML (Vectorisé avant la boucle pour la rapidité)
    print("🤖 Interrogation du modèle Machine Learning...")
    model = joblib.load(MODEL_FILE)
    features = [
        'zscore_entry', 'zscore_mom_1h', 'zscore_mom_4h', 
        'aave_ret_1h', 'eth_ret_1h', 'aave_vol_4h', 'eth_vol_4h'
    ]
    df_test['ml_prob'] = model.predict_proba(df_test[features])[:, 1]

    # 5. Moteur de Simulation Event-Driven (Boucle sans Look-ahead)
    print("⏳ Démarrage de la simulation temporelle...")
    
    capital = 100.0  # Portefeuille de départ
    equity_curve = []
    
    in_position = False
    direction = 0  # 1 = Long Spread (Long AAVE, Short ETH), -1 = Short Spread
    entry_price_aave = 0
    entry_price_eth = 0
    
    # Frais Binance VIP 0 (0.05% par patte = 0.10% par entrée, 0.10% par sortie)
    FEE_PER_LEG = 0.0005 
    
    trades_executed = 0
    trades_rejected_by_ml = 0

    # Optimisation Numpy pour la boucle
    aave_prices = df_test['AAVE'].values
    eth_prices = df_test['ETH'].values
    zscores = df_test['zscore_entry'].values
    ml_probs = df_test['ml_prob'].values
    timestamps = df_test.index

    for i in range(len(df_test) - 1): # -1 pour permettre l'exécution sur i+1
        # État au temps 't'
        current_z = zscores[i]
        prob = ml_probs[i]
        
        # --- GESTION DE LA SORTIE ---
        if in_position:
            # Condition de sortie : Retour au Z-Score = 0
            if (direction == 1 and current_z >= 0) or (direction == -1 and current_z <= 0):
                # Exécution de la sortie au prix d'OUVERTURE de la bougie suivante (t+1)
                exit_aave = aave_prices[i+1]
                exit_eth = eth_prices[i+1]
                
                # Calcul du rendement de chaque patte
                ret_aave = (exit_aave - entry_price_aave) / entry_price_aave
                ret_eth = (entry_price_eth - exit_eth) / entry_price_eth # Inversé car Short
                
                # Inversion des rendements si on était Short Spread
                if direction == -1:
                    ret_aave = -ret_aave
                    ret_eth = -ret_eth
                    
                # Frais de sortie sur les deux pattes
                gross_pnl = (ret_aave + ret_eth) / 2 # PnL moyen de la position couverte
                net_pnl = gross_pnl - (FEE_PER_LEG * 2)
                
                capital = capital * (1 + net_pnl)
                in_position = False
                direction = 0

        # --- GESTION DE L'ENTRÉE ---
        elif not in_position:
            # Le Moteur Statistique détecte une anomalie
            if current_z <= -2.0 or current_z >= 2.0:
                # Le Filtre ML prend sa décision (Seuil > 60%)
                if prob > 0.60:
                    in_position = True
                    direction = 1 if current_z <= -2.0 else -1
                    
                    # Exécution de l'entrée au prix d'OUVERTURE de la bougie suivante (t+1)
                    entry_price_aave = aave_prices[i+1]
                    entry_price_eth = eth_prices[i+1]
                    
                    # Déduction des frais d'entrée immédiats
                    capital = capital * (1 - (FEE_PER_LEG * 2))
                    trades_executed += 1
                else:
                    trades_rejected_by_ml += 1

        # Enregistrement de la valeur du portefeuille pour le graphique
        # (Note: Le PnL latent n'est pas mark-to-market ici pour simplifier la courbe)
        equity_curve.append(capital)

    # Ajout du dernier point
    equity_curve.append(capital)
    df_test['Equity'] = equity_curve

    # 6. Résultats
    pnl_net = capital - 100.0
    print("\n" + "="*70)
    print("📊 RÉSULTATS DU BACKTEST HYBRIDE (Out-Of-Sample)")
    print("="*70)
    print(f"💰 PnL Net (Après frais)   : {pnl_net:+.2f} %")
    print(f"✅ Trades Exécutés (Bons)  : {trades_executed}")
    print(f"🚫 Pièges Évités par le ML : {trades_rejected_by_ml}")
    print("="*70)
    
    # ==========================================
    # 7. CALCULS AVANCÉS ET GRAPHIQUE EN %
    # ==========================================
    import matplotlib.ticker as mtick

    # Conversion en Pourcentage
    df_test['Equity_Pct'] = (df_test['Equity'] / 100.0 - 1) * 100
    
    # Calcul du Drawdown (Baisse depuis le sommet)
    df_test['Peak'] = df_test['Equity_Pct'].cummax()
    df_test['Drawdown'] = df_test['Equity_Pct'] - df_test['Peak']
    max_dd = df_test['Drawdown'].min()

    print(f"📉 Maximum Drawdown        : {max_dd:.2f} %")
    print("="*70)

    # Création du Graphique à 2 étages
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # --- Haut : Courbe de Performance (%) ---
    ax1.plot(df_test.index, df_test['Equity_Pct'], color='#10b981', linewidth=2, label='ML-Filtered StatArb')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('Out-Of-Sample Performance (Net of Fees)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Performance Cumulative', fontsize=12)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=2)) # Formatage en % précis
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.1)

    # --- Bas : Courbe de Drawdown (%) ---
    ax2.fill_between(df_test.index, df_test['Drawdown'], 0, color='#ef4444', alpha=0.3)
    ax2.plot(df_test.index, df_test['Drawdown'], color='#ef4444', linewidth=1.5, label=f'Max Drawdown: {max_dd:.2f}%')
    ax2.set_ylabel('Drawdown', fontsize=12)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=2))
    ax2.legend(loc='lower left')
    ax2.grid(alpha=0.1)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()