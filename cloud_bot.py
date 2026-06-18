import os
import threading
from flask import Flask
from scriptsv2.livev2 import run_trading_bot 

app = Flask(__name__)

@app.route('/')
def home():
    return "🟢 Trading Bot Polymarket (V2) is running !"

@app.route('/csv')
def show_csv():
    """Affiche le contenu du fichier d'historique directement sur le web."""
    file_path = "data/trade_history.csv"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return f"<pre>{content}</pre>"
    else:
        return "⏳ Le fichier CSV n'existe pas encore."

def start_bot_background():
    print("🚀 Lancement du Bot en tâche de fond...")
    run_trading_bot()

# --- LE CORRECTIF EST ICI ---
# On lance le thread directement dans le corps du script, 
# sans attendre la condition __main__. Gunicorn sera forcé de l'exécuter.
bot_thread = threading.Thread(target=start_bot_background)
bot_thread.daemon = True
bot_thread.start()
# ----------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)