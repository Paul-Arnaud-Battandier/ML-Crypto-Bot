"""
train_meta.py
-------------
Meta-labelling (Lopez de Prado, Ch.10) avec Random Forest.

Principe :
  - Le modèle primaire (LGBM) prédit la DIRECTION : UP (1) ou DOWN (0)
  - Le meta-model (RF) prédit si ce signal mérite d'être joué : OUI (1) / NON (0)
  - Label méta = 1 si LGBM avait raison, 0 si LGBM avait tort
  - Le RF utilise la proba LGBM + features Polymarket + features Binance sélectionnées

Résultat : on ne trade que quand les deux modèles sont alignés.
           → moins de trades, mais meilleure précision.

Référence : Lopez de Prado, "Advances in Financial Machine Learning", Ch.10
"""

import json
import logging
import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, brier_score_loss,
                             log_loss, precision_score, recall_score,
                             roc_auc_score)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).parents[2]
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "model"

FEATURES_CSV  = DATA_DIR / "final_features_v2.csv"
LGBM_PKL      = MODEL_DIR / "lgbm_model.pkl"
META_PKL      = MODEL_DIR / "meta_model.pkl"
META_JSON     = MODEL_DIR / "meta_meta.json"

NON_FEATURES = {
    "timestamp", "open", "high", "low", "close", "volume",
    "index", "label", "meta_label",
}

# ---------------------------------------------------------------------------
# Features Polymarket simulées pour l'entraînement historique
# (en prod, ces valeurs viennent du snapshot live)
# En backtest on ne les a pas → on les simule de façon neutre
# ---------------------------------------------------------------------------
POLY_FEATURE_DEFAULTS = {
    "poly_yes_prob"   : 0.50,
    "poly_no_prob"    : 0.50,
    "poly_skew"       : 0.00,
    "poly_spread_yes" : 0.05,
    "poly_spread_no"  : 0.05,
    "poly_imb_yes"    : 0.00,
    "poly_imb_no"     : 0.00,
    "poly_net_imb"    : 0.00,
}


# =============================================================================
# 1. CONSTRUCTION DU DATASET META
# =============================================================================

def build_meta_dataset(
    features_df: pd.DataFrame,
    lgbm_payload: dict,
) -> pd.DataFrame:
    """
    Construit le dataset d'entraînement du meta-model.

    Pour chaque observation :
      - On récupère la proba LGBM (proba_up)
      - On calcule le meta_label = 1 si LGBM avait raison
      - On ajoute les features Polymarket (neutres en backtest)

    Args:
        features_df  : DataFrame complet de features (avec colonne 'label')
        lgbm_payload : dict retourné par load_model() du LGBM

    Returns:
        DataFrame avec colonnes [meta_features..., meta_label]
    """
    model       = lgbm_payload["model"]
    feature_cols = lgbm_payload["feature_cols"]

    # Préparer X pour le LGBM
    df = features_df.copy()
    df = df.dropna(subset=["label"])

    X_lgbm = df[feature_cols].fillna(df[feature_cols].median())
    y_true = df["label"].astype(int)

    # Probas LGBM sur tout le dataset (in-sample — pour le meta c'est ok)
    # En prod on utilisera les probas OOS du CV mais pour simplifier
    # on utilise les probas directes — le RF apprend à corriger le LGBM
    probas_lgbm = model.predict_proba(X_lgbm)[:, 1]
    preds_lgbm  = (probas_lgbm >= 0.5).astype(int)

    # Meta-label : 1 si LGBM avait raison, 0 sinon
    meta_labels = (preds_lgbm == y_true.values).astype(int)

    logger.info(f"Meta-labels | LGBM correct : {meta_labels.mean():.3f} "
                f"({meta_labels.sum()}/{len(meta_labels)})")

    # --- Features du meta-model ---
    # Sélection des features Binance les plus informatives pour le meta
    binance_meta_features = [
        "proba_lgbm",       # confiance du modèle primaire (ajoutée ci-dessous)
        "entropy_15m",      # désordre du marché → signal fiable si bas
        "vol_ratio",        # volatilité court/long → breakout ou calme
        "ofi_15m",          # order flow imbalance Binance
        "ofi_5m",
        "vwap_dev_15m",     # déviation VWAP → momentum ou mean-reversion
        "rsi_14",
        "autocorr_1",       # mean-reversion signal
        "vol_gk",           # volatilité Garman-Klass
        "tf15_ret_15m_bar", # return de la bougie 15m précédente
        "tf5_roc_3bar_5m",
    ]

    meta_df = pd.DataFrame(index=df.index)
    meta_df["proba_lgbm"] = probas_lgbm

    # Features Binance disponibles
    for col in binance_meta_features[1:]:   # skip proba_lgbm déjà ajoutée
        if col in df.columns:
            meta_df[col] = df[col].values
        else:
            logger.warning(f"Feature manquante : {col}")

    # Features Polymarket — neutres en backtest historique
    # En prod, ces valeurs viennent du snapshot live au moment du trade
    for feat, default_val in POLY_FEATURE_DEFAULTS.items():
        meta_df[feat] = default_val   # neutre : marché à 50/50, pas de biais

    meta_df["meta_label"] = meta_labels

    # Imputation
    meta_df = meta_df.fillna(meta_df.median())

    return meta_df


# =============================================================================
# 2. PURGED K-FOLD POUR LE META-MODEL
# =============================================================================

class PurgedKFold:
    """Identique à train_lgbm.py — reproduit ici pour indépendance du module."""

    def __init__(self, n_splits: int = 5, purge: int = 30, embargo: int = 5):
        self.n_splits = n_splits
        self.purge    = purge
        self.embargo  = embargo

    def split(self, X, y=None, groups=None):
        n         = len(X)
        fold_size = n // self.n_splits

        for k in range(self.n_splits):
            test_start  = k * fold_size
            test_end    = test_start + fold_size if k < self.n_splits - 1 else n
            purge_start = max(0, test_start - self.purge)
            embargo_end = min(n, test_end + self.embargo)

            train_idx = list(range(0, purge_start)) + list(range(embargo_end, n))
            test_idx  = list(range(test_start, test_end))

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            yield np.array(train_idx), np.array(test_idx)


# =============================================================================
# 3. ENTRAÎNEMENT META-MODEL
# =============================================================================

def train_meta_model(
    meta_df: pd.DataFrame,
    n_splits: int = 5,
) -> tuple[RandomForestClassifier, dict]:
    """
    Entraîne le Random Forest meta-model avec Purged K-Fold.

    Le RF est délibérément simple (pas trop de profondeur) pour rester
    interprétable et éviter l'overfit sur les patterns Polymarket.

    Returns:
        (model_final, metrics_dict)
    """
    feature_cols = [c for c in meta_df.columns if c != "meta_label"]
    X = meta_df[feature_cols]
    y = meta_df["meta_label"]

    logger.info(f"Meta-dataset : {X.shape[0]} samples | {X.shape[1]} features | "
                f"balance={y.mean():.3f}")

    rf_params = {
        "n_estimators"    : 200,
        "max_depth"       : 4,        # peu profond → moins d'overfit
        "min_samples_leaf": 20,       # stabilité
        "max_features"    : "sqrt",
        "class_weight"    : "balanced",
        "random_state"    : 42,
        "n_jobs"          : -1,
    }

    cv = PurgedKFold(n_splits=n_splits, purge=30)

    oos_probas = np.full(len(y), np.nan)
    oos_labels = np.full(len(y), np.nan)

    logger.info(f"Purged K-Fold Meta CV ({n_splits} folds)...")

    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_tr, y_tr)

        probas = rf.predict_proba(X_te)[:, 1]
        oos_probas[test_idx] = probas
        oos_labels[test_idx] = y_te.values

        preds = (probas >= 0.5).astype(int)
        logger.info(f"  Fold {fold+1}/{n_splits} | "
                    f"ACC={accuracy_score(y_te, preds):.4f} | "
                    f"AUC={roc_auc_score(y_te, probas):.4f} | "
                    f"Precision={precision_score(y_te, preds, zero_division=0):.4f}")

    # --- Métriques OOS ---
    mask  = ~np.isnan(oos_probas)
    oos_p = oos_probas[mask]
    oos_l = oos_labels[mask]
    preds = (oos_p >= 0.5).astype(int)

    # Taux de trades filtrés (combien de signaux LGBM on garde)
    n_trades    = preds.sum()
    trade_rate  = preds.mean()

    metrics = {
        "oos_accuracy"  : float(accuracy_score(oos_l, preds)),
        "oos_auc"       : float(roc_auc_score(oos_l, oos_p)),
        "oos_precision" : float(precision_score(oos_l, preds, zero_division=0)),
        "oos_recall"    : float(recall_score(oos_l, preds, zero_division=0)),
        "oos_logloss"   : float(log_loss(oos_l, oos_p)),
        "trade_rate"    : float(trade_rate),   # % de signaux LGBM conservés
        "n_trades"      : int(n_trades),
        "n_samples"     : int(mask.sum()),
    }

    logger.info(f"\n{'='*55}")
    logger.info(f"META OOS | ACC={metrics['oos_accuracy']:.4f} | "
                f"AUC={metrics['oos_auc']:.4f} | "
                f"Precision={metrics['oos_precision']:.4f}")
    logger.info(f"Trade rate : {trade_rate:.2%} des signaux LGBM conservés "
                f"({n_trades}/{mask.sum()})")
    logger.info(f"{'='*55}")

    # Modèle final sur tout le dataset
    rf_final = RandomForestClassifier(**rf_params)
    rf_final.fit(X, y)

    return rf_final, metrics


# =============================================================================
# 4. PIPELINE PRINCIPAL
# =============================================================================

def run_meta_training(
    features_csv: Path = FEATURES_CSV,
    lgbm_path   : Path = LGBM_PKL,
    output_path : Path = META_PKL,
) -> Optional[RandomForestClassifier]:
    """
    Pipeline complet meta-labelling :
      1. Charge features CSV + modèle LGBM
      2. Génère les meta-labels
      3. Entraîne le RF
      4. Sauvegarde

    Returns:
        RandomForestClassifier entraîné, ou None si erreur.
    """
    # --- Chargement LGBM ---
    if not lgbm_path.exists():
        logger.error(f"LGBM introuvable : {lgbm_path} — lance d'abord train_lgbm.py")
        return None

    with open(lgbm_path, "rb") as f:
        lgbm_payload = pickle.load(f)
    logger.info(f"LGBM chargé | AUC OOS={lgbm_payload['metrics'].get('oos_auc', 'N/A'):.4f}")

    # --- Chargement features ---
    if not features_csv.exists():
        logger.error(f"Features CSV introuvable : {features_csv}")
        return None

    df = pd.read_csv(features_csv, parse_dates=["timestamp"])

    # Labels primaires (direction BTC)
    close  = df["close"].values
    labels = np.full(len(close), np.nan)
    for i in range(len(close) - 15):
        labels[i] = 1.0 if close[i + 15] > close[i] else 0.0
    df["label"] = labels
    df = df.dropna(subset=["label"])

    # --- Build meta dataset ---
    logger.info("Construction du dataset meta...")
    meta_df = build_meta_dataset(df, lgbm_payload)

    # --- Entraînement ---
    rf_model, metrics = train_meta_model(meta_df)

    # Importance features
    feature_cols = [c for c in meta_df.columns if c != "meta_label"]
    importances  = pd.DataFrame({
        "feature"   : feature_cols,
        "importance": rf_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    logger.info("\nTop features meta-model :")
    for _, row in importances.iterrows():
        logger.info(f"  {row['feature']:<25} {row['importance']:.4f}")

    # --- Sauvegarde ---
    payload = {
        "model"       : rf_model,
        "feature_cols": feature_cols,
        "metrics"     : metrics,
        "poly_defaults": POLY_FEATURE_DEFAULTS,
    }
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)
    logger.info(f"Meta-model sauvegardé : {output_path}")

    with open(META_JSON, "w") as f:
        json.dump({"metrics": metrics, "feature_cols": feature_cols,
                   "importances": importances.to_dict(orient="records")}, f, indent=2)

    return rf_model


# =============================================================================
# 5. INFÉRENCE LIVE — combinaison LGBM + RF méta
# =============================================================================

def load_meta_model(path: Path = META_PKL) -> Optional[dict]:
    if not path.exists():
        logger.error(f"Meta-model introuvable : {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_combined(
    binance_features : pd.Series,
    poly_snapshot    : dict,
    lgbm_payload     : dict,
    meta_payload     : dict,
    min_lgbm_conf    : float = 0.55,
    min_meta_prob    : float = 0.55,
) -> dict:
    """
    Décision finale combinant LGBM + meta RF + Polymarket.

    Logique :
      1. LGBM prédit la direction et sa proba
      2. Si proba LGBM < min_lgbm_conf → skip (signal trop faible)
      3. RF meta prédit si le signal est fiable
      4. Si proba meta < min_meta_prob → skip
      5. Sinon → TRADE dans la direction LGBM

    Args:
        binance_features : pd.Series de build_live_features()
        poly_snapshot    : dict de get_market_snapshot()
        lgbm_payload     : dict de load_model() (LGBM)
        meta_payload     : dict de load_meta_model()
        min_lgbm_conf    : seuil minimum de confiance LGBM pour considérer le signal
        min_meta_prob    : seuil minimum de confiance meta pour trader

    Returns:
        dict : {
            trade       : bool    — on rentre en trade ?
            direction   : str     — "UP" / "DOWN" / "SKIP"
            side        : str     — "YES" / "NO" / "SKIP" (côté Polymarket)
            lgbm_proba  : float   — proba LGBM
            meta_proba  : float   — proba meta (fiabilité du signal)
            reason      : str     — pourquoi on trade ou skip
        }
    """
    from scriptsv2.training.train_lgbm import predict as lgbm_predict

    # --- 1. Signal LGBM ---
    lgbm_result = lgbm_predict(binance_features, lgbm_payload)
    proba_up    = lgbm_result["proba_up"]
    direction   = "UP" if lgbm_result["prediction"] == 1 else "DOWN"

    # --- 2. Filtre confiance LGBM ---
    if lgbm_result["confidence"] < (min_lgbm_conf - 0.5) * 2:
        return {
            "trade"     : False,
            "direction" : direction,
            "side"      : "SKIP",
            "lgbm_proba": proba_up,
            "meta_proba": None,
            "reason"    : f"LGBM trop incertain (conf={lgbm_result['confidence']:.3f})",
        }

    # --- 3. Construction features meta ---
    from scriptsv2.data.polymarket_feed import extract_poly_features
    poly_feats = extract_poly_features(poly_snapshot) if poly_snapshot else POLY_FEATURE_DEFAULTS

    meta_feature_cols = meta_payload["feature_cols"]
    meta_values = {}

    # Proba LGBM
    meta_values["proba_lgbm"] = proba_up

    # Features Binance sélectionnées
    for col in meta_feature_cols:
        if col in binance_features.index:
            meta_values[col] = binance_features[col]
        elif col in poly_feats:
            meta_values[col] = poly_feats[col]
        else:
            meta_values[col] = meta_payload["poly_defaults"].get(col, 0.0)

    # Injection features Polymarket réelles (écrasent les defaults)
    for k, v in poly_feats.items():
        if k in meta_feature_cols and v is not None:
            meta_values[k] = v

    X_meta = np.array([meta_values.get(c, 0.0) for c in meta_feature_cols]).reshape(1, -1)
    meta_proba = float(meta_payload["model"].predict_proba(X_meta)[0, 1])

    # --- 4. Filtre meta ---
    if meta_proba < min_meta_prob:
        return {
            "trade"     : False,
            "direction" : direction,
            "side"      : "SKIP",
            "lgbm_proba": proba_up,
            "meta_proba": meta_proba,
            "reason"    : f"Meta RF rejette le signal (meta_proba={meta_proba:.3f})",
        }

    # --- 5. Vérification liquidité Polymarket ---
    spread = poly_snapshot.get("yes_spread") if poly_snapshot else None
    if spread is not None and spread > 0.10:
        return {
            "trade"     : False,
            "direction" : direction,
            "side"      : "SKIP",
            "lgbm_proba": proba_up,
            "meta_proba": meta_proba,
            "reason"    : f"Spread Polymarket trop élevé ({spread:.3f}) — liquidité insuffisante",
        }

    # --- 6. GO TRADE ---
    side = "YES" if direction == "UP" else "NO"

    return {
        "trade"     : True,
        "direction" : direction,
        "side"      : side,
        "lgbm_proba": proba_up,
        "meta_proba": meta_proba,
        "reason"    : f"Signal confirmé | LGBM={proba_up:.3f} | Meta={meta_proba:.3f}",
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logger.info("=== Entraînement Meta-model ===")
    rf = run_meta_training()

    if rf:
        logger.info("\n=== Test predict_combined (simulation) ===")
        import sys
        sys.path.insert(0, str(ROOT))

        with open(LGBM_PKL, "rb") as f:
            lgbm_pay = pickle.load(f)
        with open(META_PKL, "rb") as f:
            meta_pay = pickle.load(f)

        # Features Binance dummy
        dummy_binance = pd.Series(
            np.random.randn(len(lgbm_pay["feature_cols"])),
            index=lgbm_pay["feature_cols"]
        )

        # Snapshot Polymarket simulé
        dummy_poly = {
            "yes_mid"      : 0.52,
            "no_mid"       : 0.48,
            "yes_spread"   : 0.04,
            "no_spread"    : 0.04,
            "yes_imbalance": 0.05,
            "no_imbalance" : -0.03,
        }

        result = predict_combined(dummy_binance, dummy_poly, lgbm_pay, meta_pay)

        print(f"\n{'='*50}")
        print(f"Trade       : {'✅ OUI' if result['trade'] else '❌ NON'}")
        print(f"Direction   : {result['direction']}")
        print(f"Côté Poly   : {result['side']}")
        print(f"LGBM proba  : {result['lgbm_proba']:.4f}")
        print(f"Meta proba  : {result['meta_proba']}")
        print(f"Raison      : {result['reason']}")
        print(f"{'='*50}")