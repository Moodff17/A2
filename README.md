# Phishing URL Detection (COS30049 Assignment 2 - AI4Cyber)

Team **Anthropyc** (Group 4): Nicholas Kan, Krishaga Hasithya, Gimhana Wijethunga.

A user pastes a web address and the system returns a **trust score (0-100)**, a **Safe / Suspicious / Dangerous**
band, and **plain-language reasons**. The URL is only ever treated as text - it is **never opened, resolved or
contacted**, so the tool runs fully offline.

The project has two machine-learning parts:
* **Classification** - benign (0) vs phishing (1).
* **Clustering** - groups the phishing URLs *without using labels* and describes each group by how its average
  feature values differ from the overall phishing average.

---
## 1. Set up the environment (conda)

```bash
conda env create -f environment.yml        # creates the "phishing-url" environment
conda activate phishing-url
```
Without conda: `python -m venv .venv`, activate it, then `pip install -r requirements.txt`.

## 2. Get the data

Raw data is not stored in the repo (size). Put the files here:

| File | Where | Required? |
|---|---|---|
| Kaggle *Malicious URLs dataset* (`malicious_phish.csv`, columns `url,type`) - https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset | `Dataset/Raw/` | **Yes** |
| Tranco top-sites list, file name starting `tranco` (`.csv`, bare domains, treated as benign) | `Dataset/Raw/Extra/` | optional |
| OpenPhish feed, file name starting `openphish` (`.txt`, one URL per line, treated as phishing) | `Dataset/Raw/Extra/` | optional |
| PhishTank export, file name starting `phishtank` (`.csv` with a `url` column, treated as phishing) | `Dataset/Raw/Extra/` | optional |
| Any other CSV with `url` and `type` columns (`benign`/`phishing`) | `Dataset/Raw/Extra/` | optional |

The **processed dataset** (features + labels + train/test split) is already provided in `Dataset/Processed/`
inside the submission zip, so steps 3-4 can be skipped if you only want to train/predict.

## 3. Process the data (raw URLs -> features)

```bash
python Main.py process
```
Creates `Dataset/Processed/processed_urls.csv`. Steps: load -> cap class sizes -> parse/normalise every URL
(`src/URL_Parser.py`) -> extract features -> remove unparseable rows, duplicates and conflicting labels ->
stratified 80/20 train/test split. `defacement` and `malware` rows are not used for training; they are stored with
`split = holdout` for the unseen-category test.

Optional analysis: `python Main.py explore` (tables + figures in `reports/`) and `python Main.py baseline`
(sanity baselines incl. a shuffled-label leakage check).

## 4. Train the classifiers

```bash
python Main.py train
```
Trains Logistic Regression, Random Forest, HistGradientBoosting, XGBoost and LightGBM (the last two only if installed),
picks the best by **validation** F1, reports every model on the **test** split, and saves models to `models/`
(`best_model.joblib`, `feature_columns.json`). Results: `reports/model_comparison.csv` and figures in `reports/figures/`.

## 5. Cluster the phishing URLs

```bash
python Main.py cluster            # k chosen by silhouette score
python Main.py cluster --k 5      # or force k
```
Outputs `reports/cluster_profiles.md` (composition, z-scores vs the phishing average, absent features, example URLs),
`reports/cluster_zscores.csv`, `Dataset/Processed/phishing_clusters.csv` and figures.

## 6. Evaluate robustness

```bash
python Main.py evaluate
```
Unseen categories (defacement/malware), unseen domains (grouped split), URL-shape and source breakdowns, and an
error analysis (`reports/error_analysis.csv`).

## 7. Predict for a URL (offline)

```bash
python Main.py predict "http://paypal.com.secure-login.xyz/verify?account=1" "https://www.google.com"
```
Prints the trust score, band and reasons. Run `python Main.py all` to do steps 3-6 in one go.

## 8. Tests
```bash
python -m unittest discover tests -v
```

---
## Project layout
```
Main.py                    entry point (all commands above)
src/
  config.py                paths and settings (caps, seed, thresholds)
  URL_Parser.py            cleans + parses any raw record into one consistent form
  Lexical_Features.py      character-level features
  Extract_Domains.py       hostname-structure features (subdomains, TLD, punycode, brand misuse)
  Network_Features.py      offline address-mechanics features (IP host, port, '@', shorteners)
  Feature_Pipeline.py      raw URL -> feature row (same code for training and prediction)
  Prepare_Data.py          raw datasets -> Dataset/Processed/processed_urls.csv
  Explore_Features.py      exploratory analysis
  Sanity_Baseline.py       baselines and leakage check
  Models.py                candidate models + metrics
  Train_Classifiers.py     training, comparison, saving
  Cluster_Phishing.py      label-free clustering within the phishing class
  Evaluate_Robustness.py   harder tests + error analysis
  Predict.py               score one URL, with reasons
tests/                     unit tests
docs/feature_justification.md   why each feature exists (+ candidate references)
Dataset/Raw, Dataset/Processed, models/, reports/
```

## Reproducibility
All randomness uses `RANDOM_STATE = 42` (`src/config.py`). Settings such as `BENIGN_CAP` and
`USE_SCHEME_FEATURES` are documented in that file.

## Data and references
Siddhartha, M. (2021) *Malicious URLs dataset* [Kaggle dataset]. Available at:
https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset (check author, year and access date before citing in Harvard style).
