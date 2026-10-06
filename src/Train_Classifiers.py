"""
Train_Classifiers.py - train, compare and save the classification models.

For every candidate model (Models.py):
  * fit on 85% of the TRAIN split, score on the remaining 15% "validation" split;
  * the model with the best validation F1 becomes the final model (chosen without
    looking at the test set);
  * every model is then scored once on the untouched TEST split for the report table.
Outputs: reports/model_comparison.csv, confusion-matrix and feature-importance figures,
models/*.joblib, models/feature_columns.json, models/best_model_name.txt.

Run:  python Main.py train
"""
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.Feature_Pipeline import get_feature_columns, load_processed
from src.Models import BEYOND_UNIT, available_models, make_model, score_binary


def plot_confusion(metrics, title, path):
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    plt.figure(figsize=(4, 3.6)); plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, f"{v:,}", ha="center", va="center",
                 color="white" if v > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["benign", "phishing"]); plt.yticks([0, 1], ["benign", "phishing"])
    plt.xlabel("predicted"); plt.ylabel("actual"); plt.title(title, fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_importance(model, feats, title, path):
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        imp = np.abs(est.coef_[0])
    else:
        return
    order = np.argsort(imp)[-15:]
    plt.figure(figsize=(6.5, 4.5)); plt.barh(np.array(feats)[order], imp[order], color="#4c78a8")
    plt.title(title, fontsize=9); plt.xlabel("importance"); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def main():
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_processed()
    feats = get_feature_columns(df)
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    sub_tr, val = train_test_split(train, test_size=0.15, stratify=train["label"],
                                   random_state=config.RANDOM_STATE)
    print(f"train {len(sub_tr):,} | validation {len(val):,} | test {len(test):,} | "
          f"features {len(feats)}")

    rows, fitted = [], {}
    for name in available_models():
        model = make_model(name)
        t0 = time.time(); model.fit(sub_tr[feats], sub_tr["label"]); fit_s = time.time() - t0
        v = score_binary(val["label"], model.predict(val[feats]), model.predict_proba(val[feats])[:, 1])
        t0 = time.time(); prob = model.predict_proba(test[feats])[:, 1]
        ms_per_url = (time.time() - t0) / len(test) * 1000
        t = score_binary(test["label"], (prob >= 0.5).astype(int), prob)
        rows.append({"model": name, "beyond_unit": name in BEYOND_UNIT, "val_f1": v["f1"],
                     "test_accuracy": t["accuracy"], "test_precision": t["precision"],
                     "test_recall": t["recall"], "test_f1": t["f1"], "test_roc_auc": t["roc_auc"],
                     "test_false_positive_rate": t["false_positive_rate"],
                     "fit_seconds": fit_s, "predict_ms_per_url": ms_per_url,
                     "_t": t})
        fitted[name] = model
        joblib.dump(model, config.MODELS_DIR / f"{name.replace(' ', '_')}.joblib")
        print(f"  {name:22s} val F1 {v['f1']:.4f} | test F1 {t['f1']:.4f} | fit {fit_s:.1f}s")

    table = pd.DataFrame(rows)
    best = table.sort_values("val_f1", ascending=False).iloc[0]
    best_name = best["model"]
    table.drop(columns="_t").round(4).to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)

    joblib.dump(fitted[best_name], config.BEST_MODEL_FILE)
    config.BEST_MODEL_NAME_FILE.write_text(best_name)
    config.FEATURE_LIST_FILE.write_text(json.dumps(feats, indent=2))

    plot_confusion(best["_t"], f"{best_name} (test set)", config.FIGURES_DIR / "confusion_matrix_best.png")
    plot_importance(fitted[best_name], feats, f"{best_name}: top features",
                    config.FIGURES_DIR / "feature_importance_best.png")
    print(f"\nBest model by validation F1: {best_name}")
    print(table.drop(columns=["_t", "val_f1"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
