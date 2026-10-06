"""
Prepare_Data.py - raw datasets -> one clean, feature-ready CSV.

Pipeline
  1. Load the base Kaggle dataset (columns: url, type) from Dataset/Raw/.
  2. Load OPTIONAL additional datasets from Dataset/Raw/Extra/ (see README for accepted
     file types). Different sources have different shapes - everything is pushed through
     URL_Parser.parse_url so they end up in one consistent format.
  3. Cap class sizes (config.py) so the data is balanced enough and the CSV stays small.
  4. Extract features (Feature_Pipeline.py), drop unparseable rows.
  5. De-duplicate on the canonical URL and drop URLs that appear with conflicting labels.
  6. Task framing: benign (0) vs phishing (1). 'defacement' and 'malware' are NOT used for
     training - they are kept as label -1, split 'holdout', for the unseen-category test.
  7. Stratified train/test split for the benign/phishing rows. Save to Dataset/Processed/.

Run:  python Main.py process
"""
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.Feature_Pipeline import build_feature_frame, get_feature_columns


# ------------------------------------------------------------------ loading
def find_base_dataset():
    """Find the Kaggle CSV in Dataset/Raw (any *.csv that has 'url' and 'type' columns)."""
    for path in sorted(config.RAW_DIR.glob("*.csv")):
        try:
            cols = pd.read_csv(path, nrows=0).columns.str.lower().tolist()
        except Exception:
            continue
        if "url" in cols and "type" in cols:
            return path
    raise FileNotFoundError(
        f"No base dataset found in {config.RAW_DIR}. Download 'malicious_phish.csv' from the "
        "Kaggle 'malicious-urls-dataset' and put it in Dataset/Raw/ (see README).")


def load_base_dataset():
    path = find_base_dataset()
    print(f"[load] base dataset: {path.name}")
    df = pd.read_csv(path, usecols=lambda c: c.lower() in ("url", "type"))
    df.columns = [c.lower() for c in df.columns]
    df = df.dropna(subset=["url", "type"])
    df["type"] = df["type"].astype(str).str.strip().str.lower()
    df["source"] = "kaggle_base"
    return df


def load_extra_sources(cap=config.EXTRA_CAP_PER_SOURCE):
    """Optional extra data. Recognised by file name:
         tranco*.csv      top-sites list (rank,domain) - BARE DOMAINS, labelled benign
         phishtank*.csv   PhishTank export with a 'url' column - labelled phishing
         openphish*.txt   OpenPhish feed, one URL per line - labelled phishing
         anything*.csv    any other CSV with 'url' and 'type' columns (types as in base data)
       Each is capped at `cap` rows so no source swamps the base dataset."""
    frames = []
    if not config.EXTRA_DIR.exists():
        return pd.DataFrame(columns=["url", "type", "source"])
    for path in sorted(config.EXTRA_DIR.iterdir()):
        name = path.name.lower()
        df = None
        if name.startswith("tranco") and name.endswith(".csv"):
            raw = pd.read_csv(path, header=None)
            col = raw.iloc[:, 1] if raw.shape[1] >= 2 else raw.iloc[:, 0]
            df = pd.DataFrame({"url": col.astype(str)})
            df = df[df["url"].str.lower() != "domain"]
            df["type"], df["source"] = "benign", "tranco"
        elif name.startswith("phishtank") and name.endswith(".csv"):
            raw = pd.read_csv(path)
            raw.columns = [c.lower() for c in raw.columns]
            df = pd.DataFrame({"url": raw["url"].astype(str)})
            df["type"], df["source"] = "phishing", "phishtank"
        elif name.startswith("openphish") and name.endswith(".txt"):
            lines = [l.strip() for l in path.read_text(errors="replace").splitlines() if l.strip()]
            df = pd.DataFrame({"url": lines})
            df["type"], df["source"] = "phishing", "openphish"
        elif name.endswith(".csv"):
            raw = pd.read_csv(path)
            raw.columns = [c.lower() for c in raw.columns]
            if {"url", "type"} <= set(raw.columns):
                df = raw[["url", "type"]].copy()
                df["type"] = df["type"].astype(str).str.strip().str.lower()
                df["source"] = path.stem.lower()
        if df is None:
            continue
        if len(df) > cap:
            df = df.sample(cap, random_state=config.RANDOM_STATE)
        print(f"[load] extra source {path.name}: {len(df):,} rows ({df['type'].iloc[0]})")
        frames.append(df)
    if not frames:
        print("[load] no additional datasets found in Dataset/Raw/Extra/ - using base data only")
        return pd.DataFrame(columns=["url", "type", "source"])
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ main pipeline
def cap_classes(base, benign_cap, holdout_cap):
    """Down-sample benign and held-out categories (phishing is always kept in full)."""
    parts = []
    for category, group in base.groupby("type"):
        cap = None
        if category == "benign":
            cap = benign_cap
        elif category in config.HOLDOUT_CATEGORIES:
            cap = holdout_cap
        elif category != "phishing":
            continue                                   # unknown category: ignore
        if cap and len(group) > cap:
            group = group.sample(cap, random_state=config.RANDOM_STATE)
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def main(benign_cap=config.BENIGN_CAP, holdout_cap=config.HOLDOUT_CAP_PER_CATEGORY):
    base = cap_classes(load_base_dataset(), benign_cap, holdout_cap)
    extra = load_extra_sources()
    raw = pd.concat([base, extra], ignore_index=True)       # base first => wins on duplicates
    print(f"[process] rows before cleaning: {len(raw):,}")

    # --- feature extraction (also validates / canonicalises each URL)
    feats = build_feature_frame(raw["url"])
    df = raw.loc[feats.index, ["type", "source"]].rename(columns={"type": "category"}).join(feats)
    print(f"[process] unparseable rows dropped: {len(raw) - len(df):,}")

    # --- duplicates and conflicting labels
    conflicts = df.groupby("canonical_url")["category"].nunique()
    conflict_urls = conflicts[conflicts > 1].index
    df = df[~df["canonical_url"].isin(conflict_urls)]
    before = len(df)
    df = df.drop_duplicates("canonical_url", keep="first")
    print(f"[process] conflicting-label URLs removed: {len(conflict_urls):,}; "
          f"duplicates removed: {before - len(df):,}")

    # --- task framing: benign vs phishing, the rest held out
    df["label"] = df["category"].map({"benign": config.BENIGN_LABEL,
                                      "phishing": config.PHISHING_LABEL}).fillna(-1).astype(int)
    df["split"] = "holdout"
    binary = df.index[df["label"] >= 0]
    train_idx, test_idx = train_test_split(
        binary, test_size=config.TEST_SIZE, stratify=df.loc[binary, "label"],
        random_state=config.RANDOM_STATE)
    df.loc[train_idx, "split"] = "train"
    df.loc[test_idx, "split"] = "test"

    # --- save
    feature_cols = [c for c in df.columns if c not in config.NON_FEATURE_COLUMNS]
    ordered = ["canonical_url", "registered_domain", "source", "category", "label", "split",
               "had_scheme"] + feature_cols
    out = df[ordered].copy()
    out[feature_cols] = out[feature_cols].round(4)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.PROCESSED_FILE, index=False)

    print(f"\n[process] saved {len(out):,} rows x {len(feature_cols)} features -> {config.PROCESSED_FILE}")
    print(out.groupby(["split", "category"]).size().to_string())
    print("\nrows per source:\n" + out["source"].value_counts().to_string())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benign-cap", type=int, default=config.BENIGN_CAP)
    ap.add_argument("--holdout-cap", type=int, default=config.HOLDOUT_CAP_PER_CATEGORY)
    a = ap.parse_args()
    main(a.benign_cap, a.holdout_cap)
