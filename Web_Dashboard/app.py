"""
app.py — StatArb Dashboard
Flask app déployé sur Render.
Lit les données depuis Supabase et affiche le dashboard.
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


@app.route('/')
def index():
    # ── Equity curve (500 derniers points) ────────────────────
    equity_raw = sb_get('live_equity', 'order=id.asc&limit=500')

    # ── Trades récents ─────────────────────────────────────────
    trades = sb_get('live_trades', 'order=id.desc&limit=20')

    # ── Régime actuel ──────────────────────────────────────────
    regime_data = sb_get('regime_history', 'order=id.desc&limit=1')
    regime = regime_data[0] if regime_data else None

    # ── Stats sur les trades fermés ────────────────────────────
    all_exits = sb_get('live_trades', 'action=eq.EXIT&order=id.desc&limit=500')
    pnl_list  = [t['pnl_pct'] for t in all_exits if t.get('pnl_pct') is not None]
    wins      = [p for p in pnl_list if p > 0]

    # Stop losses
    sl_trades  = [t for t in all_exits if t.get('exit_reason', '').startswith('STOP')]
    tp_trades  = [t for t in all_exits if t.get('exit_reason') == 'TP']

    stats = {
        'total_trades': len(pnl_list),
        'win_rate'    : round(len(wins) / len(pnl_list) * 100, 1) if pnl_list else 0,
        'total_pnl'   : round(sum(pnl_list) * 100, 2) if pnl_list else 0,
        'avg_pnl'     : round(sum(pnl_list) / len(pnl_list) * 100, 3) if pnl_list else 0,
        'tp_count'    : len(tp_trades),
        'sl_count'    : len(sl_trades),
    }

    # ── État actuel ─────────────────────────────────────────────
    current = equity_raw[-1] if equity_raw else {
        'equity_usdt': 0,
        'position_status': 'UNKNOWN',
        'unrealized_pnl_pct': 0,
    }

    # ── Données graphique ───────────────────────────────────────
    chart_labels = []
    chart_values = []
    for e in equity_raw:
        ts = e.get('timestamp', '')
        chart_labels.append(ts[5:16] if len(ts) > 15 else ts)  # MM-DD HH:MM
        chart_values.append(round(e.get('equity_usdt', 0), 2))

    return render_template('index.html',
        regime=regime,
        trades=trades,
        stats=stats,
        current=current,
        chart_labels=chart_labels,
        chart_values=chart_values,
        last_update=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    )


@app.route('/api/status')
def api_status():
    """Endpoint JSON pour checks externes"""
    regime_data = sb_get('regime_history', 'order=id.desc&limit=1')
    equity_data = sb_get('live_equity', 'order=id.desc&limit=1')
    return jsonify({
        'status'   : 'online',
        'regime'   : regime_data[0].get('regime') if regime_data else None,
        'equity'   : equity_data[0].get('equity_usdt') if equity_data else None,
        'timestamp': datetime.now().isoformat(),
    })


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)