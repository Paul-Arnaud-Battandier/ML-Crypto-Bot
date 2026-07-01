"""
build_dataset.py
----------------
Script one-shot : construit final_features_v2.csv depuis binance_1m_raw.csv.

À lancer UNE FOIS avant l'entraînement des modèles.

Usage :
    python build_dataset.py

Ce que ça fait :
  1. Charge data/binance_1m_raw.csv (ton historique Binance)
  2. Backfill si données manquantes (complète jusqu'à aujourd'hui)
  3. Calcule toutes les features (feature_pipeline.py)
  4. Sauvegarde data/final_features_v2.csv

Ensuite :
    python scriptsv2/models/train_lgbm.py
    python scriptsv2/models/train_meta.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# --- Path setup ---
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("build_dataset")

DATA_DIR     = ROOT / "data"
RAW_CSV      = DATA_DIR / "binance_1m_raw.csv"
FEATURES_CSV = DATA_DIR / "final_features_v2.csv"


def main():
    logger.info("=" * 60)
    logger.info("BUILD DATASET — one-shot pipeline")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Chargement / backfill Binance
    # ------------------------------------------------------------------
    from scripts.data.binance_feed import backfill

    logger.info("Étape 1 : Chargement et mise à jour Binance 1m...")
    df_raw = backfill(days=90)

    if df_raw is None or df_raw.empty:
        logger.error("Impossible de charger les données Binance. Abandon.")
        sys.exit(1)

    logger.info(f"Données brutes : {len(df_raw)} bougies 1m | "
                f"{df_raw['timestamp'].min()} → {df_raw['timestamp'].max()}")

    # Vérification qualité
    n_missing = df_raw.isnull().sum().sum()
    if n_missing > 0:
        logger.warning(f"{n_missing} valeurs manquantes dans le raw — on les droppe")
        df_raw = df_raw.dropna()

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    logger.info("Étape 2 : Calcul des features...")
    from scripts.features.feature_pipeline import build_features

    df_features = build_features(df_raw)

    logger.info(f"Features calculées : {df_features.shape[1]} colonnes | "
                f"{df_features.shape[0]} lignes")

    # ------------------------------------------------------------------
    # 3. Nettoyage
    # ------------------------------------------------------------------
    logger.info("Étape 3 : Nettoyage...")

    # Colonnes avec trop de NaN (warm-up des fenêtres glissantes)
    nan_ratio    = df_features.isnull().mean()
    cols_bad     = nan_ratio[nan_ratio > 0.10].index.tolist()
    cols_to_keep = [c for c in df_features.columns if c not in cols_bad]

    if cols_bad:
        logger.warning(f"Colonnes droppées (>10% NaN) : {cols_bad}")

    df_clean = df_features[cols_to_keep].copy()

    # Supprimer les premières lignes (warm-up des features)
    # On garde seulement les lignes avec moins de 5% de NaN
    row_nan_ratio = df_clean.isnull().mean(axis=1)
    df_clean      = df_clean[row_nan_ratio < 0.05].reset_index(drop=True)

    logger.info(f"Après nettoyage : {df_clean.shape[0]} lignes | "
                f"{df_clean.shape[1]} colonnes")

    # ------------------------------------------------------------------
    # 4. Sauvegarde
    # ------------------------------------------------------------------
    logger.info(f"Étape 4 : Sauvegarde → {FEATURES_CSV}")
    DATA_DIR.mkdir(exist_ok=True)
    df_clean.to_csv(FEATURES_CSV, index=False)

    logger.info("=" * 60)
    logger.info("✅ Dataset prêt !")
    logger.info(f"   Fichier : {FEATURES_CSV}")
    logger.info(f"   Shape   : {df_clean.shape}")
    logger.info(f"   Période : {df_clean['timestamp'].min()} → {df_clean['timestamp'].max()}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Prochaines étapes :")
    logger.info("  python scriptsv2/models/train_lgbm.py")
    logger.info("  python scriptsv2/models/train_meta.py")


if __name__ == "__main__":
    main()