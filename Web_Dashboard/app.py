"""
app.py — Trading Bot Dashboard
Flask app déployé sur Render.
Lit les données depuis Supabase (StatArb + Funding Carry) et affiche le dashboard.
"""

import os
import requests
from flask import Flask, render_template, jsonify
from datetime import datetime

app = Flask(__name__)

SUPABASE_URL = os.getenv('SUPABASE_URL', '')
SUPABASE_KEY = os.getenv('SUPABASE_KEY', '')

# ── Lancement des bots en arrière-plan ──────────────────────────
# ⚠️ Mettre RUN_BACKGROUND_JOBS=true UNIQUEMENT sur Render.
#    En local, laisser à false (ou absent) pour éviter un double bot
#    si live_bot.py tourne déjà dans un terminal séparé.
# StatArb/Funding tournent ici (pas sur GitHub Actions) car Binance
# bloque les IP des runners GitHub (erreur 451) — Render fonctionne.
# Chaque bot tourne en sous-processus isolé (voir background_jobs.py)
# pour éviter l'accumulation mémoire qui causait les OOM précédents.
if os.getenv('RUN_BACKGROUND_JOBS', 'false').lower() == 'true':
    from background_jobs import start_background_jobs
    start_background_jobs()


def sb_get(table, params=''):
    """Appel REST Supabase — retourne [] si erreur"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return []
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/{table}?{params}",
            headers={
                'apikey': SUPABASE_KEY,
                'Authorization': f'Bearer {SUPABASE_KEY}',
                'Range': '0-999',
            },
            timeout=5
        )
        return r.json() if r.status_code in (200, 206) else []
    except Exception as e:
        print(f"Supabase error: {e}")
        return []


def sb_get_all(table, params='', batch_size=1000):
    """
    Comme sb_get, mais pagine automatiquement via l'en-tête Range pour
    récupérer TOUTE la table, quelle que soit sa taille — plus de plafond
    fixe (ex: limit=3000) qui coupait l'historique aux N dernières lignes.
    Utilisé pour l'equity curve : on veut le tracé complet depuis la toute
    première position prise par le bot, pas juste les derniers mois.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return []
    all_rows = []
    offset = 0
    while True:
        try:
            r = requests.get(
                f"{SUPABASE_URL}/rest/v1/{table}?{params}",
                headers={
                    'apikey': SUPABASE_KEY,
                    'Authorization': f'Bearer {SUPABASE_KEY}',
                    'Range': f'{offset}-{offset + batch_size - 1}',
                },
                timeout=15
            )
        except Exception as e:
            print(f"Supabase error: {e}")
            break
        if r.status_code not in (200, 206):
            break
        batch = r.json()
        if not batch:
            break
        all_rows.extend(batch)
        if len(batch) < batch_size:
            break  # dernier lot -> on a tout récupéré
        offset += batch_size
    return all_rows


def build_synthetic_equity(rows, capital_reference, current_contrib_fn):
    """
    Reconstruit une courbe de capital propre à UNE stratégie, à partir de
    son propre P&L réalisé + latent — indépendamment du solde brut du
    compte Binance (equity_usdt), qui est PARTAGÉ entre StatArb et Funding
    sur le même wallet Futures. Une position ouverte par l'un des deux
    bots réserve de la marge sur ce wallet commun, ce qui fait "bouger"
    le solde lu par l'AUTRE bot alors qu'il n'a rien fait — découvert le
    04/09 quand l'equity StatArb chutait pendant qu'il était FLAT, à
    cause d'une position Funding fraîchement ouverte sur le même compte.

    current_contrib_fn(row) doit renvoyer la contribution $ (réalisée +
    latente) de la position EN COURS à cette ligne précise (0 si FLAT) —
    différente pour StatArb (unrealized_pnl_usdt) et Funding
    (funding_collected_usd + unrealized_pnl_usd).
    """
    values = []
    realized_cum = 0.0
    prev_status  = None
    prev_contrib = 0.0
    for row in rows:
        status  = row.get('position_status', 'FLAT') or 'FLAT'
        contrib = current_contrib_fn(row) if status != 'FLAT' else 0.0
        if prev_status not in (None, 'FLAT') and status == 'FLAT':
            # La position précédente vient de se fermer -> on fige
            # définitivement son P&L dans le cumul réalisé.
            realized_cum += prev_contrib
        values.append(round(capital_reference + realized_cum + contrib, 2))
        prev_status, prev_contrib = status, contrib
    return values


def get_statarb_data():
    """Récupère toutes les données StatArb (equity, trades, stats)"""
    # Historique complet depuis la toute première position du bot, pas
    # juste les N derniers points : order=id.asc + pagination automatique
    # (sb_get_all) pour ne jamais couper la courbe, quelle que soit la
    # taille de la table.
    equity_raw = sb_get_all('live_equity', 'order=id.asc')
    trades     = sb_get('live_trades', 'order=id.desc&limit=20')

    all_exits = sb_get_all('live_trades', 'action=eq.EXIT&order=id.asc')
    pnl_list  = [t['pnl_pct'] for t in all_exits if t.get('pnl_pct') is not None]
    wins      = [p for p in pnl_list if p > 0]
    # Wins/Losses basés sur le vrai P&L, pas sur le nom du déclencheur de sortie —
    # un exit "SIGNAL_EXIT" (Z revenu à 0) peut être perdant après frais/hedge imparfait,
    # et ce n'est pas pour autant un Stop-Loss (les seuils SL ne se sont pas déclenchés).
    win_trades  = [p for p in pnl_list if p > 0]
    loss_trades = [p for p in pnl_list if p <= 0]

    stats = {
        'total_trades'    : len(pnl_list),
        'win_rate'        : round(len(wins) / len(pnl_list) * 100, 1) if pnl_list else 0,
        # Somme des rendements % de chaque trade, calculés sur le notionnel
        # exposé par trade (~$100 : 50$ AAVE + 50$ ETH) — PAS sur le capital
        # total du compte. Utile pour juger la qualité de la stratégie
        # indépendamment de la taille du compte, mais ne pas confondre avec
        # le rendement réel du portefeuille (voir 'portfolio_return_pct').
        'cumulative_trade_pnl': round(sum(pnl_list) * 100, 2) if pnl_list else 0,
        'avg_pnl'         : round(sum(pnl_list) / len(pnl_list) * 100, 3) if pnl_list else 0,
        'win_count'       : len(win_trades),
        'loss_count'      : len(loss_trades),
    }

    current = equity_raw[-1] if equity_raw else {
        'equity_usdt': 0, 'position_status': 'UNKNOWN', 'unrealized_pnl_pct': 0,
    }
    current = dict(current)  # copie — on va remplacer equity_usdt affiché

    chart_labels, chart_values = [], []
    capital_reference = round(equity_raw[0].get('equity_usdt', 0), 2) if equity_raw else 0
    synthetic_values = build_synthetic_equity(
        equity_raw, capital_reference,
        current_contrib_fn=lambda row: row.get('unrealized_pnl_usdt', 0) or 0
    )
    for e in equity_raw:
        ts = e.get('timestamp', '')
        chart_labels.append(ts[5:16] if len(ts) > 15 else ts)
    chart_values = synthetic_values
    if chart_values:
        current['equity_usdt'] = chart_values[-1]  # cohérent avec la courbe —
                                                     # sinon la carte "Capital"
                                                     # afficherait encore le
                                                     # solde brut contaminé.

    # Rendement propre à StatArb (P&L réalisé+latent de SES trades), pas
    # l'évolution du solde brut du compte (contaminé par le funding bot
    # qui partage le même wallet Futures).
    if len(chart_values) >= 2 and capital_reference:
        stats['portfolio_return_pct'] = round(
            (chart_values[-1] - chart_values[0]) / capital_reference * 100, 3
        )
    else:
        stats['portfolio_return_pct'] = 0.0

    return {
        'trades': trades, 'stats': stats, 'current': current,
        'chart_labels': chart_labels, 'chart_values': chart_values,
    }


def get_funding_data():
    """Récupère toutes les données Funding Carry (equity, trades, stats)"""
    # Même fix que StatArb : historique complet depuis la première position,
    # via pagination automatique plutôt qu'un plafond fixe.
    equity_raw = sb_get_all('funding_equity', 'order=id.asc')
    trades     = sb_get('funding_trades', 'order=id.desc&limit=20')

    all_exits = sb_get_all('funding_trades', 'action=eq.EXIT&order=id.asc')
    total_funding_list = [t['total_funding_collected_usd'] for t in all_exits
                           if t.get('total_funding_collected_usd') is not None]

    current = equity_raw[-1] if equity_raw else {
        'equity_usdt': 0, 'position_status': 'FLAT', 'symbol': '',
        'funding_collected_usd': 0, 'unrealized_pnl_usd': 0,
    }
    current = dict(current)

    chart_labels, chart_values = [], []
    capital_reference = round(equity_raw[0].get('equity_usdt', 0), 2) if equity_raw else 0
    chart_values = build_synthetic_equity(
        equity_raw, capital_reference,
        current_contrib_fn=lambda row: (row.get('funding_collected_usd', 0) or 0)
                                      + (row.get('unrealized_pnl_usd', 0) or 0)
    )
    for e in equity_raw:
        ts = e.get('timestamp', '')
        chart_labels.append(ts[5:16] if len(ts) > 15 else ts)
    if chart_values:
        current['equity_usdt'] = chart_values[-1]

    stats = {
        'total_closed_positions'   : len(all_exits),
        'total_funding_all_time'   : round(sum(total_funding_list), 4) if total_funding_list else 0,
        'current_funding_collected': round(current.get('funding_collected_usd', 0) or 0, 4),
        'avg_funding_per_position' : round(sum(total_funding_list) / len(total_funding_list), 4) if total_funding_list else 0,
    }

    return {
        'trades': trades, 'stats': stats, 'current': current,
        'chart_labels': chart_labels, 'chart_values': chart_values,
    }


def get_regime_distribution(limit=500):
    """
    Compte combien de fois chaque régime a été observé sur les <limit>
    derniers relevés horaires (limit=500 ≈ 20 jours d'historique).
    Permet de voir concrètement quelle stratégie a été active le plus
    souvent — utile pour comprendre pourquoi Funding Carry (actif en
    HIGH_VOL/NEUTRAL) peut sembler silencieux si MEAN_REV domine.
    """
    rows = sb_get('regime_history', f'order=id.desc&limit={limit}')
    counts = {'MEAN_REV': 0, 'TRENDING': 0, 'HIGH_VOL': 0, 'NEUTRAL': 0}
    for r in rows:
        reg = r.get('regime')
        if reg in counts:
            counts[reg] += 1
    total = sum(counts.values())
    return {
        'labels' : list(counts.keys()),
        'counts' : list(counts.values()),
        'total'  : total,
        'pct'    : {k: round(v / total * 100, 1) if total else 0 for k, v in counts.items()},
    }


@app.route('/')
def index():
    statarb  = get_statarb_data()
    funding  = get_funding_data()

    regime_data = sb_get('regime_history', 'order=id.desc&limit=1')
    regime = regime_data[0] if regime_data else None
    regime_dist = get_regime_distribution()

    # ── Résumé global (somme des deux stratégies) ─────────────
    global_capital = (statarb['current'].get('equity_usdt') or 0) + \
                      (funding['current'].get('equity_usdt') or 0)

    return render_template('index.html',
        regime=regime,
        regime_dist=regime_dist,
        statarb=statarb,
        funding=funding,
        global_capital=global_capital,
        last_update=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    )


@app.route('/api/status')
def api_status():
    """Endpoint JSON pour checks externes"""
    regime_data  = sb_get('regime_history', 'order=id.desc&limit=1')
    equity_data  = sb_get('live_equity', 'order=id.desc&limit=1')
    funding_data = sb_get('funding_equity', 'order=id.desc&limit=1')
    return jsonify({
        'status'        : 'online',
        'regime'        : regime_data[0].get('regime') if regime_data else None,
        'statarb_equity': equity_data[0].get('equity_usdt') if equity_data else None,
        'funding_equity': funding_data[0].get('equity_usdt') if funding_data else None,
        'timestamp'     : datetime.now().isoformat(),
    })


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)