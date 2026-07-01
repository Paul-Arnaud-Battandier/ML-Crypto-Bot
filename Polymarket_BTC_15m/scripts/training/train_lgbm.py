"""
train_lgbm.py
-------------
Entraînement du modèle primaire LightGBM pour prédire BTC up/down 15m.

Pipeline :
  1. Labels binaires  — close[t+15] > close[t] → 1 (up) / 0 (down)
  2. Purged K-Fold    — évite le data leakage des features à fenêtre glissante
  3. LGBM             — classification binaire, proba calibrée
  4. Sauvegarde       — model/lgbm_model.pkl + metadata JSON

Référence : Lopez de Prado, Ch.7 (Cross-Validation in Finance)
"""

import json
import logging
import pickle
import warnings
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, brier_score_loss,
                             log_loss, roc_auc_score)
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).parents[2]
DATA_DIR   = ROOT / "data"
MODEL_DIR  = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES_CSV = DATA_DIR / "final_features_v2.csv"
MODEL_PKL    = MODEL_DIR / "lgbm_model.pkl"
META_JSON    = MODEL_DIR / "lgbm_meta.json"

# ---------------------------------------------------------------------------
# Colonnes à exclure du feature set
# ---------------------------------------------------------------------------
NON_FEATURES = {
    "timestamp", "open", "high", "low", "close", "volume",
    "index", "label", "ret_15m_future",
}


# =============================================================================
# 1. LABELS
# =============================================================================

def make_labels(df: pd.DataFrame, horizon: int = 15) -> pd.Series:
    """
    Label binaire : le close dans `horizon` bougies 1m sera-t-il supérieur
    au close actuel ?

    label = 1 si close[t + horizon] > close[t]  (BTC monte)
    label = 0 sinon                               (BTC baisse)

    On utilise le close 1m et on resample ensuite pour aligner
    sur les décisions toutes les 15 minutes.

    Args:
        df      : DataFrame avec colonnes [timestamp, close]
        horizon : nombre de bougies 1m dans le futur (défaut=15)

    Returns:
        Series de labels (float, NaN pour les dernières lignes sans futur)
    """
    close = df["close"].values
    labels = np.full(len(close), np.nan)

    for i in range(len(close) - horizon):
        labels[i] = 1.0 if close[i + horizon] > close[i] else 0.0

    return pd.Series(labels, index=df.index, name="label")


# =============================================================================
# 2. PURGED K-FOLD (Lopez de Prado, Ch.7)
# =============================================================================

class PurgedKFold:
    """
    K-Fold avec purge des observations qui chevauchent train/test.

    Problème classique en finance :
      - Les features utilisent des fenêtres glissantes (ex: vol sur 30 bougies)
      - Sans purge, les features du test "voient" des données du train → leakage
      - Solution : purger les N premières observations du fold de test
                   (N = taille max de la fenêtre glissante)

    Args:
        n_splits : nombre de folds (défaut=5)
        purge    : nombre de bougies à purger entre train et test (défaut=30)
        embargo  : bougies supplémentaires à embargo après le test (défaut=5)
    """

    def __init__(self, n_splits: int = 5, purge: int = 30, embargo: int = 5):
        self.n_splits = n_splits
        self.purge    = purge
        self.embargo  = embargo

    def split(self, X, y=None, groups=None):
        n = len(X)
        fold_size = n // self.n_splits

        for k in range(self.n_splits):
            # Indices test
            test_start = k * fold_size
            test_end   = test_start + fold_size if k < self.n_splits - 1 else n

            # Indices train : tout sauf [test - purge : test_end + embargo]
            purge_start = max(0, test_start - self.purge)
            embargo_end = min(n, test_end + self.embargo)

            train_idx = list(range(0, purge_start)) + list(range(embargo_end, n))
            test_idx  = list(range(test_start, test_end))

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            yield np.array(train_idx), np.array(test_idx)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# =============================================================================
# 3. MODÈLE LGBM
# =============================================================================

def build_lgbm_params() -> dict:
    """Hyperparamètres LGBM pour classification binaire."""
    return {
        "objective"        : "binary",
        "metric"           : "binary_logloss",
        "boosting_type"    : "gbdt",
        "n_estimators"     : 500,
        "learning_rate"    : 0.02,
        "num_leaves"       : 31,
        "max_depth"        : 5,
        "min_child_samples": 30,       # évite l'overfit sur petits noeuds
        "feature_fraction" : 0.7,      # sous-échantillonnage des features
        "bagging_fraction" : 0.8,
        "bagging_freq"     : 5,
        "reg_alpha"        : 0.1,      # L1
        "reg_lambda"       : 0.1,      # L2
        "random_state"     : 42,
        "verbose"          : -1,
        "n_jobs"           : -1,
    }


# =============================================================================
# 4. ENTRAÎNEMENT AVEC PURGED CV
# =============================================================================

def train_with_purged_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    purge: int = 30,
) -> tuple[lgb.LGBMClassifier, dict]:
    """
    Entraîne LGBM avec Purged K-Fold et retourne le modèle final + métriques OOS.

    La stratégie :
      1. Purged K-Fold pour évaluer les performances out-of-sample
      2. Entraînement final sur TOUT le dataset pour la prod

    Returns:
        (model_final, metrics_dict)
    """
    cv     = PurgedKFold(n_splits=n_splits, purge=purge)
    params = build_lgbm_params()

    oos_probas = np.full(len(y), np.nan)
    oos_labels = np.full(len(y), np.nan)

    logger.info(f"Purged K-Fold CV ({n_splits} folds, purge={purge})...")

    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)],
        )

        probas = model.predict_proba(X_te)[:, 1]
        oos_probas[test_idx] = probas
        oos_labels[test_idx] = y_te.values

        fold_acc = accuracy_score(y_te, (probas >= 0.5).astype(int))
        fold_auc = roc_auc_score(y_te, probas)
        logger.info(f"  Fold {fold+1}/{n_splits} | ACC={fold_acc:.4f} | AUC={fold_auc:.4f} "
                    f"| best_iter={model.best_iteration_}")

    # --- Métriques OOS globales ---
    mask = ~np.isnan(oos_probas)
    oos_p = oos_probas[mask]
    oos_l = oos_labels[mask]

    metrics = {
        "oos_accuracy" : float(accuracy_score(oos_l, (oos_p >= 0.5).astype(int))),
        "oos_auc"      : float(roc_auc_score(oos_l, oos_p)),
        "oos_logloss"  : float(log_loss(oos_l, oos_p)),
        "oos_brier"    : float(brier_score_loss(oos_l, oos_p)),
        "n_samples"    : int(mask.sum()),
        "n_features"   : X.shape[1],
        "class_balance": float(y.mean()),
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"OOS GLOBAL | ACC={metrics['oos_accuracy']:.4f} | "
                f"AUC={metrics['oos_auc']:.4f} | "
                f"LogLoss={metrics['oos_logloss']:.4f} | "
                f"Brier={metrics['oos_brier']:.4f}")
    logger.info(f"{'='*50}")

    # --- Modèle final sur tout le dataset ---
    logger.info("Entraînement final sur 100% des données...")
    model_final = lgb.LGBMClassifier(**params)
    model_final.fit(X, y, callbacks=[lgb.log_evaluation(-1)])

    return model_final, metrics


# =============================================================================
# 5. IMPORTANCE DES FEATURES
# =============================================================================

def get_feature_importance(model: lgb.LGBMClassifier, feature_names: list) -> pd.DataFrame:
    """Retourne un DataFrame trié par importance (gain)."""
    importance = model.feature_importances_
    df = pd.DataFrame({
        "feature"   : feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df


# =============================================================================
# 6. PIPELINE PRINCIPAL
# =============================================================================

def run_training(
    features_csv: Path = FEATURES_CSV,
    model_output: Path = MODEL_PKL,
    n_splits: int = 5,
    min_samples: int = 500,
) -> Optional[lgb.LGBMClassifier]:
    """
    Pipeline complet : CSV features → modèle entraîné sauvegardé.

    Args:
        features_csv : chemin vers le CSV de features (build_features output)
        model_output : où sauvegarder le modèle
        n_splits     : nombre de folds CV
        min_samples  : minimum de samples requis pour entraîner

    Returns:
        Le modèle entraîné, ou None si données insuffisantes.
    """

    # --- Chargement ---
    if not features_csv.exists():
        logger.error(f"CSV introuvable : {features_csv}")
        return None

    logger.info(f"Chargement : {features_csv}")
    df = pd.read_csv(features_csv, parse_dates=["timestamp"])
    logger.info(f"Shape brut : {df.shape}")

    # --- Labels ---
    df["label"] = make_labels(df, horizon=15)
    df = df.dropna(subset=["label"])

    if len(df) < min_samples:
        logger.error(f"Données insuffisantes : {len(df)} samples (min={min_samples})")
        return None

    # --- Features ---
    feature_cols = [c for c in df.columns if c not in NON_FEATURES]
    X = df[feature_cols].copy()
    y = df["label"].astype(int)

    # Drop des colonnes avec trop de NaN (warm-up)
    nan_ratio = X.isna().mean()
    cols_to_drop = nan_ratio[nan_ratio > 0.05].index.tolist()
    if cols_to_drop:
        logger.warning(f"Drop {len(cols_to_drop)} colonnes avec >5% NaN : {cols_to_drop}")
        X = X.drop(columns=cols_to_drop)

    # Imputation des NaN restants par la médiane
    X = X.fillna(X.median())

    logger.info(f"Dataset final : {X.shape[0]} samples | {X.shape[1]} features | "
                f"balance={y.mean():.3f} (1=up)")

    # --- CV + entraînement ---
    model, metrics = train_with_purged_cv(X, y, n_splits=n_splits)

    # --- Importance features ---
    importance_df = get_feature_importance(model, X.columns.tolist())
    logger.info("\nTop 10 features :")
    for _, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']:<30} {row['importance']:.0f}")

    # --- Sauvegarde modèle ---
    model_output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model"        : model,
        "feature_cols" : X.columns.tolist(),
        "metrics"      : metrics,
        "importance"   : importance_df.to_dict(orient="records"),
    }
    with open(model_output, "wb") as f:
        pickle.dump(payload, f)
    logger.info(f"Modèle sauvegardé : {model_output}")

    # --- Metadata JSON (lisible sans pickle) ---
    meta = {
        "feature_cols": X.columns.tolist(),
        "metrics"     : metrics,
        "top10_features": importance_df.head(10).to_dict(orient="records"),
        "n_splits"    : n_splits,
    }
    with open(META_JSON, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata : {META_JSON}")

    return model


# =============================================================================
# INFÉRENCE — utilisé par le bot live
# =============================================================================

def load_model(model_path: Path = MODEL_PKL) -> Optional[dict]:
    """
    Charge le modèle depuis le pkl.

    Returns:
        dict : {model, feature_cols, metrics} ou None.
    """
    if not model_path.exists():
        logger.error(f"Modèle introuvable : {model_path}")
        return None

    with open(model_path, "rb") as f:
        payload = pickle.load(f)

    logger.info(f"Modèle chargé | AUC OOS={payload['metrics'].get('oos_auc', 'N/A'):.4f}")
    return payload


def predict(features: pd.Series, model_payload: dict) -> dict:
    """
    Prédit la direction BTC à partir d'un vecteur de features live.

    Args:
        features      : pd.Series (une ligne de build_live_features)
        model_payload : dict retourné par load_model()

    Returns:
        dict : {
            prediction : 0 ou 1,
            proba_up   : probabilité que BTC monte,
            proba_down : probabilité que BTC baisse,
            confidence : abs(proba_up - 0.5) * 2  ∈ [0, 1]
        }
    """
    feature_cols = model_payload["feature_cols"]
    model        = model_payload["model"]

    # Aligner les features avec celles attendues par le modèle
    X = features.reindex(feature_cols).fillna(0).values.reshape(1, -1)

    proba_up   = float(model.predict_proba(X)[0, 1])
    proba_down = 1.0 - proba_up
    prediction = 1 if proba_up >= 0.5 else 0
    confidence = abs(proba_up - 0.5) * 2   # 0 = incertain, 1 = certain

    return {
        "prediction" : prediction,
        "proba_up"   : round(proba_up, 4),
        "proba_down" : round(proba_down, 4),
        "confidence" : round(confidence, 4),
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    import sys

    # Test synthétique si pas de CSV réel
    if not FEATURES_CSV.exists():
        logger.warning(f"CSV introuvable, génération de données synthétiques...")

        from pathlib import Path
        import sys
        sys.path.insert(0, str(ROOT))
        from scripts.features.feature_pipeline import build_features

        np.random.seed(42)
        n = 2000
        price = 100_000 * np.cumprod(1 + np.random.normal(0, 0.001, n))
        ts    = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
        df_raw = pd.DataFrame({
            "timestamp": ts,
            "open"     : price * (1 + np.random.normal(0, 0.0002, n)),
            "high"     : price * (1 + np.abs(np.random.normal(0, 0.0005, n))),
            "low"      : price * (1 - np.abs(np.random.normal(0, 0.0005, n))),
            "close"    : price,
            "volume"   : np.random.lognormal(10, 1, n),
        })

        feat_df = build_features(df_raw)
        DATA_DIR.mkdir(exist_ok=True)
        feat_df.to_csv(FEATURES_CSV, index=False)
        logger.info(f"CSV synthétique sauvegardé : {FEATURES_CSV} ({len(feat_df)} lignes)")

    model = run_training()

    if model:
        logger.info("\n=== Test inférence live ===")
        payload = load_model()
        if payload:
            dummy_features = pd.Series(
                np.random.randn(len(payload["feature_cols"])),
                index=payload["feature_cols"]
            )
            result = predict(dummy_features, payload)
            print(f"\nPrédiction : {'UP ↑' if result['prediction'] == 1 else 'DOWN ↓'}")
            print(f"Proba UP   : {result['proba_up']:.4f}")
            print(f"Proba DOWN : {result['proba_down']:.4f}")
            print(f"Confiance  : {result['confidence']:.4f}")