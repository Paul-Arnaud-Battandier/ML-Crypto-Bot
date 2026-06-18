import os
import threading
from flask import Flask
# On importe ta fonction de trading (vérifie le nom de tes dossiers/fichiers)
from scriptsv2.livev2 import run_trading_bot 

app = Flask(__name__)

@app.route('/')
def home():
    """Cette page sert juste à dire à Render que le bot est en vie."""
    return "🟢 Trading Bot Polymarket (V2) is running !"

def start_bot_background():
    """Lance ton bot dans un thread séparé pour ne pas bloquer le serveur web."""
    print("🚀 Lancement du Bot en tâche de fond...")
    run_trading_bot()

if __name__ == "__main__":
    # Démarre le bot sur une piste parallèle
    bot_thread = threading.Thread(target=start_bot_background)
    bot_thread.daemon = True
    bot_thread.start()
    
    # Démarre le serveur web pour satisfaire Render
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)