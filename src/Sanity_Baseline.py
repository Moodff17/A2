"""
Sanity_Baseline.py - checks that the real models are doing something meaningful.

Compares on the SAME test split:
  * majority-class guess,
  * random guess in proportion to class sizes,
  * a one-feature model (URL length only) - how far does the simplest idea get?,
  * the same tree model trained on SHUFFLED labels - must score near chance; if it does not,
    there is leakage somewhere in the pipeline.

Run:  python Main.py baseline
"""
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.Feature_Pipeline import get_feature_columns, load_processed
from src.Models import score_binary


def main():
    df = load_processed()
    feats = get_feature_columns(df)
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    Xtr, ytr, Xte, yte = train[feats], train["label"], test[feats], test["label"]

    experiments = {
        "Majority class": (DummyClassifier(strategy="most_frequent"), feats, ytr),
        "Random (class priors)": (DummyClassifier(strategy="stratified",
                                                  random_state=config.RANDOM_STATE), feats, ytr),
        "URL length only": (LogisticRegression(max_iter=1000), ["url_length"], ytr),
        "Shuffled labels (leak check)": (DecisionTreeClassifier(max_depth=8,
                                         random_state=config.RANDOM_STATE), feats,
                                         ytr.sample(frac=1, random_state=config.RANDOM_STATE).values),
    }
    rows = []
    for name, (model, cols, y) in experiments.items():
        model.fit(Xtr[cols], y)
        pred = model.predict(Xte[cols])
        prob = model.predict_proba(Xte[cols])[:, 1]
        rows.append({"experiment": name, **score_binary(yte, pred, prob)})
    result = pd.DataFrame(rows).set_index("experiment")
    result.to_csv(config.REPORTS_DIR / "sanity_baseline.csv")
    print(result[["accuracy", "precision", "recall", "f1", "roc_auc"]].round(3).to_string())


if __name__ == "__main__":
    main()
