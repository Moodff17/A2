"""
Models.py - the candidate classifiers and the shared evaluation metrics.

Keeping model definitions in one place means Train_Classifiers.py and
Evaluate_Robustness.py always build exactly the same model.

TAUGHT vs BEYOND-UNIT: edit BEYOND_UNIT below to match what COS30049 actually taught.
The rubric rewards at least two models not taught in the unit, compared with the baselines
on the same test set, WITH reasoning (write that in the report - see docs/).
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

# Models treated as "beyond the unit" (adjust to what was actually taught!)
BEYOND_UNIT = {"XGBoost", "LightGBM", "HistGradientBoosting"}


def make_model(name):
    """Build a fresh, untrained model by name."""
    rs = config.RANDOM_STATE
    if name == "Logistic Regression":       # linear baseline; scaling matters for LR
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced"))
    if name == "Random Forest":             # non-linear baseline
        return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=rs,
                                      class_weight="balanced_subsample")
    if name == "HistGradientBoosting":      # sklearn's histogram gradient boosting
        return HistGradientBoostingClassifier(random_state=rs)
    if name == "XGBoost" and XGBClassifier is not None:
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                             colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1,
                             random_state=rs, tree_method="hist")
    if name == "LightGBM" and LGBMClassifier is not None:
        return LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=63,
                              random_state=rs, n_jobs=-1, verbose=-1)
    raise ValueError(f"Model '{name}' is unknown or its library is not installed.")


def available_models():
    names = ["Logistic Regression", "Random Forest", "HistGradientBoosting"]
    if XGBClassifier is not None:
        names.append("XGBoost")
    else:
        print("[models] xgboost not installed - skipping XGBoost (pip install xgboost)")
    if LGBMClassifier is not None:
        names.append("LightGBM")
    else:
        print("[models] lightgbm not installed - skipping LightGBM (pip install lightgbm)")
    return names


def score_binary(y_true, y_pred, y_prob=None):
    """Standard metrics. 'Positive' = phishing (1)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_positive_rate": fp / max(fp + tn, 1),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    return out
