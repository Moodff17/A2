"""
Evaluate_Robustness.py - tries to BREAK the final model (Assignment 2 "innovation in evaluation").

A single score on a random split says little, so this script runs four harder checks:

 1. UNSEEN CATEGORY: the model never saw 'defacement' or 'malware' URLs. What share does it
    still flag as suspicious? (Compared with the false-positive rate on benign test URLs.)
 2. UNSEEN DOMAINS: re-train on a split where no registered domain appears in both train
    and test (GroupShuffleSplit). Random splits let the same site leak into both sides.
 3. DIFFERENT SHAPES: scores split by URL shape (bare domain vs has path) and by data source.
 4. ERROR ANALYSIS: dumps false positives / false negatives to reports/error_analysis.csv and
    prints how their features differ from correctly classified URLs. YOU must then read
    ~10 of each by hand and write down what they have in common.

Run:  python Main.py evaluate
"""
import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src import config
from src.Feature_Pipeline import get_feature_columns, load_processed
from src.Models import make_model, score_binary


def main():
    df = load_processed()
    feats = get_feature_columns(df)
    model = joblib.load(config.BEST_MODEL_FILE)
    name = config.BEST_MODEL_NAME_FILE.read_text().strip()
    binary = df[df["label"] >= 0]
    test = binary[binary["split"] == "test"].copy()
    test["prob"] = model.predict_proba(test[feats])[:, 1]
    test["pred"] = (test["prob"] >= 0.5).astype(int)
    results = []

    # ---- 1. unseen categories
    base = score_binary(test["label"], test["pred"], test["prob"])
    print(f"Final model: {name}.  Random-split test F1 = {base['f1']:.4f}\n")
    benign_fpr = base["false_positive_rate"]
    for cat in config.HOLDOUT_CATEGORIES:
        sub = df[df["category"] == cat]
        if len(sub) == 0:
            continue
        flagged = (model.predict_proba(sub[feats])[:, 1] >= 0.5).mean()
        print(f"[unseen category] {cat:11s}: {flagged:6.1%} flagged as phishing  "
              f"(benign false-positive rate for reference: {benign_fpr:.1%})")
        results.append({"check": "unseen_category", "subset": cat, "n": len(sub),
                        "detected_share": flagged})

    # ---- 2. unseen domains (grouped split)
    gss = GroupShuffleSplit(n_splits=1, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    tr_i, te_i = next(gss.split(binary, groups=binary["registered_domain"]))
    tr, te = binary.iloc[tr_i], binary.iloc[te_i]
    g_model = make_model(name).fit(tr[feats], tr["label"])
    gp = g_model.predict_proba(te[feats])[:, 1]
    g = score_binary(te["label"], (gp >= 0.5).astype(int), gp)
    print(f"\n[unseen domains] grouped split F1 {g['f1']:.4f} (precision {g['precision']:.3f}, "
          f"recall {g['recall']:.3f})  vs random split F1 {base['f1']:.4f}")
    results.append({"check": "grouped_by_domain", "subset": "all", "n": len(te), **g})

    # ---- 3. shape / source breakdown
    print("\n[shape and source] test-split F1 by subset:")
    subsets = {"bare domain (no path)": test["path_length"] == 0,
               "has path": test["path_length"] > 0}
    for src in test["source"].unique():
        subsets[f"source={src}"] = test["source"] == src
    for label, mask in subsets.items():
        sub = test[mask]
        if sub["label"].nunique() < 2:
            print(f"  {label:26s} n={len(sub):7,}  (only one class present - F1 not meaningful)")
            continue
        s = score_binary(sub["label"], sub["pred"], sub["prob"])
        print(f"  {label:26s} n={len(sub):7,}  F1 {s['f1']:.3f}  precision {s['precision']:.3f}  recall {s['recall']:.3f}")
        results.append({"check": "subset", "subset": label, "n": len(sub), **s})

    # ---- 4. error analysis
    fp = test[(test["label"] == 0) & (test["pred"] == 1)].sort_values("prob", ascending=False)
    fn = test[(test["label"] == 1) & (test["pred"] == 0)].sort_values("prob")
    keep = ["canonical_url", "source", "label", "prob"] + feats
    pd.concat([fp.head(100).assign(error="false_positive"),
               fn.head(100).assign(error="false_negative")])[["error"] + keep].round(3).to_csv(
        config.REPORTS_DIR / "error_analysis.csv", index=False)
    print(f"\n[error analysis] {len(fp):,} false positives, {len(fn):,} false negatives on the test split")
    for title, errors, correct in (("FALSE POSITIVES vs correct benign", fp, test[(test.label == 0) & (test.pred == 0)]),
                                   ("FALSE NEGATIVES vs correct phishing", fn, test[(test.label == 1) & (test.pred == 1)])):
        if len(errors) == 0 or len(correct) == 0:
            continue
        d = ((errors[feats].mean() - correct[feats].mean()) / test[feats].std().replace(0, 1)).sort_values(key=abs, ascending=False)
        print(f"\n  {title} - biggest standardised feature differences:")
        print(d.head(5).round(2).to_string())
        print("  examples:")
        for u in errors["canonical_url"].head(8):
            print(f"    {u[:100]}")
    pd.DataFrame(results).to_csv(config.REPORTS_DIR / "robustness_results.csv", index=False)
    print("\nSaved reports/robustness_results.csv and reports/error_analysis.csv")


if __name__ == "__main__":
    main()
