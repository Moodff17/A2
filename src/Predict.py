"""
Predict.py - score ONE URL with the trained model. Fully offline.

The URL is treated purely as text: it is parsed and measured by Feature_Pipeline.py and
never opened, resolved or contacted. Output: trust score (0-100, higher = safer), a
Safe / Suspicious / Dangerous band, and plain-language reasons for the flags.

Run:  python Main.py predict "http://paypa1.com.secure-login.xyz/verify?account=1"
"""
import json

import joblib
import pandas as pd

from src import config
from src.Feature_Pipeline import extract_features


def explain(f, canonical):
    """Plain-language reasons, driven by the same features the model sees."""
    reasons = []
    if f["is_ip_host"]:
        reasons.append("The address uses a raw IP address instead of a normal website name.")
    if f["has_punycode"] or f["has_non_ascii_host"]:
        reasons.append("The web address contains look-alike (non-English) characters.")
    if f["brand_lookalike_host"]:
        reasons.append("A part of the address looks like a misspelled version of a well-known brand.")
    if f["brand_in_subdomain"] or f["brand_in_path"]:
        reasons.append("A well-known brand name appears in the address, but the site is not that brand's real domain.")
    if f["tld_is_risky"]:
        reasons.append("It ends in a domain extension that is often abused by scammers.")
    if f["has_userinfo"]:
        reasons.append("It contains an '@' that can hide the real destination.")
    if f["url_length"] > 75:
        reasons.append(f"The address is very long ({int(f['url_length'])} characters).")
    if f["subdomain_count"] >= 3:
        reasons.append("It has many sub-sections before the main domain name.")
    if f["suspicious_keyword_count"] >= 2:
        reasons.append("It contains several words scammers use (e.g. login, verify, secure).")
    if f["nonstandard_port"]:
        reasons.append("It uses an unusual network port.")
    if f["is_shortener"]:
        reasons.append("It is a shortened link, so the real destination is hidden.")
    if f["hyphen_count"] >= 4:
        reasons.append("It contains many hyphens, a common trick in fake addresses.")
    if f["percent_encoded_count"] >= 3:
        reasons.append("It contains many encoded characters that can disguise words.")
    return reasons or ["No strong warning signs were found in the address text."]


def predict_url(url, model=None, feature_cols=None):
    """Return a dict with trust_score, band, probability_phishing and reasons."""
    result = extract_features(url)
    if result is None:
        return {"error": "That does not look like a valid URL or domain name."}
    meta, feats = result
    if model is None:
        if not config.BEST_MODEL_FILE.exists():
            raise FileNotFoundError("No trained model found - run `python Main.py train` first.")
        model = joblib.load(config.BEST_MODEL_FILE)
    if feature_cols is None:
        feature_cols = json.loads(config.FEATURE_LIST_FILE.read_text())
    row = pd.DataFrame([{c: feats.get(c, 0) for c in feature_cols}])[feature_cols]
    prob = float(model.predict_proba(row)[0, 1])
    score = round(100 * (1 - prob))
    band = ("Safe" if score >= config.SAFE_THRESHOLD
            else "Suspicious" if score >= config.SUSPICIOUS_THRESHOLD else "Dangerous")
    return {"url": url, "canonical": meta["canonical_url"], "trust_score": score, "band": band,
            "probability_phishing": round(prob, 4), "reasons": explain(feats, meta["canonical_url"])}


def main(urls):
    model = joblib.load(config.BEST_MODEL_FILE) if config.BEST_MODEL_FILE.exists() else None
    cols = json.loads(config.FEATURE_LIST_FILE.read_text()) if config.FEATURE_LIST_FILE.exists() else None
    for url in urls:
        r = predict_url(url, model, cols)
        if "error" in r:
            print(f"\n{url}\n  {r['error']}")
            continue
        print(f"\n{url}\n  Trust score: {r['trust_score']}/100  ->  {r['band']}   "
              f"(P(phishing) = {r['probability_phishing']})")
        for reason in r["reasons"]:
            print(f"   - {reason}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
