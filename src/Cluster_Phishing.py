"""
Cluster_Phishing.py - unsupervised clustering INSIDE the phishing class (Assignment 2 core task).

Rules from the brief, and how this script follows them:
  * clustering uses NO labels - only the numeric feature columns go into KMeans;
    the label column is used once, to select which rows ARE phishing;
  * because every row is phishing, every cluster has the same majority label, so each
    cluster is described by what its members have in common:
      - label/category and source composition (a sanity check, not a description),
      - cluster mean vs overall phishing mean for every feature, expressed as a z-score,
      - the 3 features that differ most in EACH direction (higher and lower),
      - features that are (almost) ABSENT in the cluster - often the strongest signal,
      - real example URLs from the cluster.

Outputs: reports/cluster_profiles.md, reports/cluster_zscores.csv,
         Dataset/Processed/phishing_clusters.csv, figures (PCA scatter, z-score heatmap).

Run:  python Main.py cluster            (or: python -m src.Cluster_Phishing --k 5)
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src import config
from src.Feature_Pipeline import get_feature_columns, load_processed


def choose_k(Xs, k_range=range(2, 9)):
    """Pick k by silhouette score on a sample (also saves an elbow/silhouette figure)."""
    rng = np.random.RandomState(config.RANDOM_STATE)
    sample = Xs[rng.choice(len(Xs), size=min(8000, len(Xs)), replace=False)]
    inertias, sils = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=5, random_state=config.RANDOM_STATE).fit(sample)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(sample, km.labels_))
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    ax[0].plot(list(k_range), inertias, "o-"); ax[0].set_title("Elbow (inertia)"); ax[0].set_xlabel("k")
    ax[1].plot(list(k_range), sils, "o-", color="#e45756"); ax[1].set_title("Silhouette"); ax[1].set_xlabel("k")
    plt.tight_layout(); plt.savefig(config.FIGURES_DIR / "cluster_choose_k.png", dpi=150); plt.close()
    print("silhouette by k:", {k: round(s, 3) for k, s in zip(k_range, sils)})
    return list(k_range)[int(np.argmax(sils))]


def describe_clusters(phish, feats, labels, scaler_std):
    """Return z-score table (clusters x features) and the markdown report text."""
    overall_mean = phish[feats].mean()
    z = phish[feats].groupby(labels).mean().sub(overall_mean).div(scaler_std)
    lines = ["# Phishing cluster profiles\n",
             f"{len(phish):,} phishing URLs, {labels.nunique()} clusters (KMeans, labels NOT used).\n",
             "z = (cluster mean - overall phishing mean) / overall std. "
             "|z| > 0.5 is a noticeable difference.\n"]
    for c in sorted(labels.unique()):
        members = phish[labels == c]
        zc = z.loc[c]
        higher = zc.sort_values(ascending=False).head(3)
        lower = zc.sort_values().head(3)
        absent = [f for f in feats if overall_mean[f] > 0.05 and members[f].mean() < 0.1 * overall_mean[f]]
        lines.append(f"\n## Cluster {c}  ({len(members):,} URLs, {len(members) / len(phish):.1%})\n")
        lines.append("Composition check - category: " + ", ".join(
            f"{k} {v:.0%}" for k, v in members["category"].value_counts(normalize=True).items())
            + " | source: " + ", ".join(
            f"{k} {v:.0%}" for k, v in members["source"].value_counts(normalize=True).items()) + "\n")
        lines.append("| feature | cluster mean | overall mean | z |\n|---|---|---|---|")
        for f in list(higher.index) + list(lower.index):
            lines.append(f"| {f} | {members[f].mean():.3g} | {overall_mean[f]:.3g} | {zc[f]:+.2f} |")
        lines.append("\nFeatures nearly ABSENT here (overall mean > 0.05 but cluster < 10% of it): "
                     + (", ".join(absent[:8]) if absent else "none") + "\n")
        lines.append("Example URLs (random sample):")
        for u in members["canonical_url"].sample(min(5, len(members)), random_state=config.RANDOM_STATE):
            lines.append(f"- `{u[:110]}`")
        lines.append("\n**Interpretation (write this yourself from the table + examples):** _TODO_\n")
    return z, "\n".join(lines)


def main(k=None):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_processed()
    feats = get_feature_columns(df)
    phish = df[df["label"] == config.PHISHING_LABEL].reset_index(drop=True)

    # drop constant columns (carry no information, break scaling)
    feats = [f for f in feats if phish[f].std() > 0]
    scaler = StandardScaler().fit(phish[feats])
    Xs = scaler.transform(phish[feats])                      # <-- the ONLY input to clustering

    k = k or choose_k(Xs)
    km = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit(Xs)
    labels = pd.Series(km.labels_, name="cluster")
    print(f"KMeans k={k}; cluster sizes:\n{labels.value_counts().sort_index().to_string()}")

    z, report = describe_clusters(phish, feats, labels, phish[feats].std())
    (config.REPORTS_DIR / "cluster_profiles.md").write_text(report, encoding="utf-8")
    z.round(3).to_csv(config.REPORTS_DIR / "cluster_zscores.csv")
    pd.DataFrame({"canonical_url": phish["canonical_url"], "cluster": labels}).to_csv(
        config.PROCESSED_DIR / "phishing_clusters.csv", index=False)

    # figures: PCA scatter + z-score heatmap of the most distinctive features
    pts = PCA(n_components=2, random_state=config.RANDOM_STATE).fit_transform(Xs)
    sel = np.random.RandomState(config.RANDOM_STATE).choice(len(pts), min(15000, len(pts)), replace=False)
    plt.figure(figsize=(6, 4.5))
    plt.scatter(pts[sel, 0], pts[sel, 1], c=labels.values[sel], s=4, cmap="tab10")
    plt.title(f"Phishing URLs, KMeans k={k} (PCA view)"); plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout(); plt.savefig(config.FIGURES_DIR / "cluster_pca.png", dpi=150); plt.close()

    top = z.abs().max().sort_values(ascending=False).head(15).index
    plt.figure(figsize=(8, 0.5 * k + 3)); plt.imshow(z[top].values, cmap="coolwarm", vmin=-2, vmax=2, aspect="auto")
    plt.xticks(range(len(top)), top, rotation=75, ha="right", fontsize=7)
    plt.yticks(range(k), [f"cluster {c}" for c in z.index]); plt.colorbar(label="z vs phishing average")
    plt.title("What makes each cluster different"); plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "cluster_heatmap.png", dpi=150); plt.close()
    print(f"Wrote reports/cluster_profiles.md - read it, then write your interpretation of each cluster.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None, help="force number of clusters")
    main(ap.parse_args().k)
