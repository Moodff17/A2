"""
config.py - one place for every path, constant and setting used by the project.

Keeping these here means no script has "magic" file names or numbers in it, and
a teammate can change a setting (e.g. the benign cap) without hunting through code.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "Dataset" / "Raw"
EXTRA_DIR = RAW_DIR / "Extra"          # optional additional datasets (Assignment 2 innovation)
PROCESSED_DIR = ROOT / "Dataset" / "Processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

PROCESSED_FILE = PROCESSED_DIR / "processed_urls.csv"     # features + labels + split column
FEATURE_LIST_FILE = MODELS_DIR / "feature_columns.json"   # exact column order the model expects
BEST_MODEL_FILE = MODELS_DIR / "best_model.joblib"
BEST_MODEL_NAME_FILE = MODELS_DIR / "best_model_name.txt"

# ---------------------------------------------------------------- reproducibility
RANDOM_STATE = 42

# ---------------------------------------------------------------- data preparation
# Task framing: binary classification, benign (0) vs phishing (1).
# 'defacement' and 'malware' rows are NOT used for training - they are kept as a
# held-out "category the model never saw" test (see Evaluate_Robustness.py).
BENIGN_LABEL = 0
PHISHING_LABEL = 1
HOLDOUT_CATEGORIES = ["defacement", "malware"]

# The base Kaggle data has ~428k benign vs ~94k phishing. Capping benign gives a
# less extreme imbalance and keeps the processed CSV a sensible size for the zip.
BENIGN_CAP = 150_000
HOLDOUT_CAP_PER_CATEGORY = 10_000
TEST_SIZE = 0.20

# Extra (additional) datasets: how many rows to keep from each, so one big list
# cannot swamp the base dataset.
EXTRA_CAP_PER_SOURCE = 50_000

# The base dataset may store some URLs with a scheme (http://) and some without.
# If the presence of a scheme happens to line up with the label, a model can "cheat"
# on that artefact instead of learning real phishing signals. Default: scheme-derived
# features are switched OFF. Explore_Features.py prints the scheme-vs-label table so
# the team can decide with evidence.
USE_SCHEME_FEATURES = False

# ---------------------------------------------------------------- traffic-light bands for the trust score
SAFE_THRESHOLD = 70         # trust score >= 70  -> "Safe"
SUSPICIOUS_THRESHOLD = 40   # 40 <= score < 70   -> "Suspicious", below 40 -> "Dangerous"

# ---------------------------------------------------------------- column bookkeeping
# Everything in the processed CSV that is NOT a model input. All other columns are features.
NON_FEATURE_COLUMNS = [
    "canonical_url", "registered_domain", "had_scheme",   # produced by URL_Parser
    "source", "category", "label", "split",               # added by Prepare_Data
]
