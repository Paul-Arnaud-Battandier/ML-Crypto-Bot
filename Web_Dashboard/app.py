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


def get_statarb_data():
    """Récupère toutes les données StatArb (equity, trades, stats)"""
    equity_raw = sb_get('live_equity', 'order=id.asc&limit=500')
    trades     = sb_get('live_trades', 'order=id.desc&limit=20')

    all_exits = sb_get('live_trades', 'action=eq.EXIT&order=id.desc&limit=500')
    pnl_list  = [t['pnl_pct'] for t in all_exits if t.get('pnl_pct') is not None]
    wins      = [p for p in pnl_list if p > 0]
    sl_trades = [t for t in all_exits if (t.get('exit_reason') or '').startswith('STOP')]
    tp_trades = [t for t in all_exits if t.get('exit_reason') == 'TP']

    stats = {
        'total_trades': len(pnl_list),
        'win_rate'    : round(len(wins) / len(pnl_list) * 100, 1) if pnl_list else 0,
        'total_pnl'   : round(sum(pnl_list) * 100, 2) if pnl_list else 0,
        'avg_pnl'     : round(sum(pnl_list) / len(pnl_list) * 100, 3) if pnl_list else 0,
        'tp_count'    : len(tp_trades),
        'sl_count'    : len(sl_trades),
    }

    current = equity_raw[-1] if equity_raw else {
        'equity_usdt': 0, 'position_status': 'UNKNOWN', 'unrealized_pnl_pct': 0,
    }

    chart_labels, chart_values = [], []
    for e in equity_raw:
        ts = e.get('timestamp', '')
        chart_labels.append(ts[5:16] if len(ts) > 15 else ts)
        chart_values.append(round(e.get('equity_usdt', 0), 2))

    return {
        'trades': trades, 'stats': stats, 'current': current,
        'chart_labels': chart_labels, 'chart_values': chart_values,
    }


def get_funding_data():
    """Récupère toutes les données Funding Carry (equity, trades, stats)"""
    equity_raw = sb_get('funding_equity', 'order=id.asc&limit=500')
    trades     = sb_get('funding_trades', 'order=id.desc&limit=20')

    all_exits = sb_get('funding_trades', 'action=eq.EXIT&order=id.desc&limit=500')
    total_funding_list = [t['total_funding_collected_usd'] for t in all_exits
                           if t.get('total_funding_collected_usd') is not None]

    current = equity_raw[-1] if equity_raw else {
        'equity_usdt': 0, 'position_status': 'FLAT', 'symbol': '',
        'funding_collected_usd': 0, 'unrealized_pnl_usd': 0,
    }

    stats = {
        'total_closed_positions'   : len(all_exits),
        'total_funding_all_time'   : round(sum(total_funding_list), 4) if total_funding_list else 0,
        'current_funding_collected': round(current.get('funding_collected_usd', 0) or 0, 4),
        'avg_funding_per_position' : round(sum(total_funding_list) / len(total_funding_list), 4) if total_funding_list else 0,
    }

    chart_labels, chart_values = [], []
    for e in equity_raw:
        ts = e.get('timestamp', '')
        chart_labels.append(ts[5:16] if len(ts) > 15 else ts)
        chart_values.append(round(e.get('equity_usdt', 0), 2))

    return {
        'trades': trades, 'stats': stats, 'current': current,
        'chart_labels': chart_labels, 'chart_values': chart_values,
    }


@app.route('/')
def index():
    statarb  = get_statarb_data()
    funding  = get_funding_data()

    regime_data = sb_get('regime_history', 'order=id.desc&limit=1')
    regime = regime_data[0] if regime_data else None

    # ── Résumé global (somme des deux stratégies) ─────────────
    global_capital = (statarb['current'].get('equity_usdt') or 0) + \
                      (funding['current'].get('equity_usdt') or 0)
    global_pnl_pct = statarb['stats']['total_pnl']  # StatArb en %, funding en $ séparé

    return render_template('index.html',
        regime=regime,
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