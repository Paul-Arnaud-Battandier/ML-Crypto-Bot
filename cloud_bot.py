import os
import threading
from flask import Flask
from scriptsv2.livev2 import run_trading_bot 

app = Flask(__name__)

@app.route('/')
def home():
    return "🟢 Trading Bot Polymarket (V2) is running !"

# --- LA NOUVELLE ROUTE MAGIQUE ---
@app.route('/csv')
def show_csv():
    """Affiche le contenu du fichier d'historique directement sur le web."""
    file_path = "data/trade_history.csv"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        # On utilise <pre> pour garder le formatage brut du texte sur la page web
        return f"<pre>{content}</pre>"
    else:
        return "⏳ Le fichier CSV n'existe pas encore. Le bot n'a pas encore fait de trade ou a été redémarré."
# ---------------------------------

def start_bot_background():
    print("🚀 Lancement du Bot en tâche de fond...")
    run_trading_bot()

if __name__ == "__main__":
    bot_thread = threading.Thread(target=start_bot_background)
    bot_thread.daemon = True
    bot_thread.start()
    
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)