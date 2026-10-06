"""
Explore_Features.py - exploratory data analysis of the processed data (report: "Data Analysis").

Prints and saves:
  * class balance and rows per source,
  * scheme-vs-label table (checks for the dataset artefact described in config.py),
  * mean/median of every feature by class (reports/feature_summary_by_class.csv),
  * figures in reports/figures/: class balance, URL length by class, standardised
    class-mean differences for the most discriminative features, feature correlation heatmap.

Run:  python Main.py explore
"""
import matplotlib
matplotlib.use("Agg")                       # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.Feature_Pipeline import get_feature_columns, load_processed


def main():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_processed()
    data = df[df["label"] >= 0]
    feats = get_feature_columns(df)
    names = {0: "benign", 1: "phishing"}

    # ---- 1. balance and sources
    print("Class balance (benign/phishing rows):")
    print(data["label"].map(names).value_counts().to_string(), "\n")
    print("Rows per source x class:")
    print(data.groupby(["source", data["label"].map(names)]).size().unstack(fill_value=0).to_string(), "\n")
    print("Scheme present (had_scheme) x class  -> if this is lopsided the model could cheat:")
    print(data.groupby([data["label"].map(names), "had_scheme"]).size().unstack(fill_value=0).to_string(), "\n")

    # ---- 2. summary table
    summary = data.groupby(data["label"].map(names))[feats].agg(["mean", "median"]).T
    summary.to_csv(config.REPORTS_DIR / "feature_summary_by_class.csv")

    # ---- 3. figures
    counts = data["label"].map(names).value_counts()
    plt.figure(figsize=(4.5, 3.5)); plt.bar(counts.index, counts.values, color=["#4c78a8", "#e45756"])
    plt.title("Class balance (modelling data)"); plt.ylabel("URLs"); plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "class_balance.png", dpi=150); plt.close()

    cap = data["url_length"].quantile(0.99)
    plt.figure(figsize=(6, 3.8))
    for lab, colour in ((0, "#4c78a8"), (1, "#e45756")):
        plt.hist(data.loc[data["label"] == lab, "url_length"].clip(upper=cap), bins=60,
                 alpha=0.6, label=names[lab], color=colour, density=True)
    plt.xlabel("URL length (characters, clipped at 99th percentile)"); plt.ylabel("density")
    plt.legend(); plt.title("URL length by class"); plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "url_length_by_class.png", dpi=150); plt.close()

    # standardised difference of class means: (mean_phish - mean_benign) / overall std
    std = data[feats].std().replace(0, np.nan)
    diff = ((data[data["label"] == 1][feats].mean() - data[data["label"] == 0][feats].mean()) / std).dropna()
    top = diff.reindex(diff.abs().sort_values(ascending=False).index).head(15)[::-1]
    plt.figure(figsize=(7, 5)); plt.barh(top.index, top.values,
                                         color=["#e45756" if v > 0 else "#4c78a8" for v in top.values])
    plt.axvline(0, color="k", lw=0.8); plt.xlabel("standardised difference (phishing - benign)")
    plt.title("Features that separate the classes most"); plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "top_feature_differences.png", dpi=150); plt.close()
    print("Top 10 discriminative features (standardised mean difference):")
    print(diff.reindex(diff.abs().sort_values(ascending=False).index).head(10).round(3).to_string(), "\n")

    corr = data[feats].corr().fillna(0).values
    plt.figure(figsize=(9, 8)); plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.xticks(range(len(feats)), feats, rotation=90, fontsize=6)
    plt.yticks(range(len(feats)), feats, fontsize=6); plt.colorbar(label="correlation")
    plt.title("Feature correlation"); plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "feature_correlation.png", dpi=150); plt.close()
    print(f"Figures saved to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
