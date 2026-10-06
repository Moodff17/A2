"""
Extract_Domains.py - features about the STRUCTURE of the hostname.

The hostname is the part of a URL an attacker cannot fully fake, and the part users
misread most. These features describe how the host is built (subdomain depth, top-level
domain, punycode, brand names in the wrong place, look-alike spellings) using only
the host STRING - no DNS, WHOIS or any other lookup (the app must stay offline).

Motivation / sources are listed in docs/feature_justification.md.
"""
import re

# Top-level domains disproportionately used for abuse in public reporting
# (Spamhaus Project "most abused TLDs"; Interisle Consulting Group phishing-landscape reports).
# This is a STATIC list, not learned from our labels, so it cannot leak the answer.
RISKY_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "work", "click", "link", "buzz",
    "icu", "site", "online", "live", "cyou", "monster", "rest", "zip", "mov", "cam", "sbs",
    "bond", "quest", "vip", "loan", "win", "review", "country", "kim", "party", "gdn", "support",
}
COMMON_TLDS = {"com", "org", "net", "edu", "gov"}

# Brands phishers imitate most (e.g. APWG / OpenPhish top-targeted lists). Used only to ask:
# "is a famous brand name present somewhere it does not belong?"
BRANDS = [
    "paypal", "apple", "icloud", "google", "gmail", "microsoft", "outlook", "office365",
    "amazon", "facebook", "instagram", "whatsapp", "netflix", "ebay", "dropbox", "linkedin",
    "twitter", "chase", "wellsfargo", "bankofamerica", "citibank", "fedex", "usps", "adobe",
    "steam", "spotify", "yahoo", "docusign", "coinbase", "binance", "metamask", "westpac",
    "commbank", "mygov", "auspost", "telstra",
]
_LEET = str.maketrans({"0": "o", "3": "e", "4": "a", "5": "s", "7": "t", "$": "s"})


def _brand_hits(text):
    """Brands present in text as a whole token, or (for names of 6+ letters) inside a token."""
    tokens = [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]
    hits = set()
    for b in BRANDS:
        if b in tokens or (len(b) >= 6 and any(b in t for t in tokens)):
            hits.add(b)
    return hits


def _edit_distance_is_one(a, b):
    """True if a and b differ by exactly one insert / delete / substitution."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def _brand_lookalike(label):
    """Typosquat / leetspeak copy of a brand: 'paypa1', 'paypall', 'g00gle', 'rnicrosoft'.
    Checked on every host label (not just the domain) so 'paypa1.com.evil.xyz' is caught."""
    if not label or label in BRANDS:
        return False
    variants = {label.translate(_LEET)}
    variants |= {v.replace("1", "l") for v in [label.translate(_LEET)]}
    variants |= {v.replace("1", "i") for v in [label.translate(_LEET)]}
    variants |= {v.replace("rn", "m").replace("vv", "w") for v in list(variants)}
    if any(v in BRANDS and v != label for v in variants):
        return True
    return len(label) >= 6 and any(len(b) >= 6 and _edit_distance_is_one(label, b) for b in BRANDS)


def domain_features(p):
    """p is a ParsedURL (valid). Returns a dict of numeric features."""
    if p.is_ip:     # an IP has no domain structure; Network_Features.py handles IP hosts
        return {k: 0 for k in (
            "subdomain_count", "host_label_count", "domain_label_length", "domain_hyphen_count",
            "domain_digit_count", "mixed_alnum_label_count", "tld_length", "tld_is_risky",
            "tld_is_common", "has_punycode", "has_non_ascii_host", "brand_in_subdomain",
            "brand_in_path", "brand_lookalike_host")}

    labels = [l for l in p.host_ascii.split(".") if l]
    tld = labels[-1] if labels else ""
    dom_brands = _brand_hits(p.domain_label)
    sub_brands = _brand_hits(p.subdomain + " " + p.userinfo)   # userinfo: "paypal.com@evil.tk"
    path_brands = _brand_hits(p.path + " " + p.query)

    return {
        # --- depth: 'paypal.com.secure.evil.xyz' hides the real domain behind extra labels
        "subdomain_count": len(p.subdomain.split(".")) if p.subdomain else 0,
        "host_label_count": len(labels),
        # --- the registrable part chosen by the attacker
        "domain_label_length": len(p.domain_label),
        "domain_hyphen_count": p.domain_label.count("-"),
        "domain_digit_count": sum(c.isdigit() for c in p.domain_label),
        # labels that mix letters and digits ('paypa1', 'g00gle') - common in look-alikes
        "mixed_alnum_label_count": sum(
            1 for l in labels if re.search(r"[a-z]", l) and re.search(r"\d", l)),
        # --- top-level domain
        "tld_length": len(tld),
        "tld_is_risky": int(tld in RISKY_TLDS),
        "tld_is_common": int(tld in COMMON_TLDS),
        # --- look-alike characters: punycode (RFC 3492) and non-ASCII hosts enable homograph attacks
        "has_punycode": int(any(l.startswith("xn--") for l in labels)),
        "has_non_ascii_host": int(any(ord(c) > 127 for c in p.host)),
        # --- brand placement: brand present but NOT as the registered domain
        "brand_in_subdomain": int(bool(sub_brands - dom_brands)),
        "brand_in_path": int(bool(path_brands - dom_brands)),
        "brand_lookalike_host": int(any(_brand_lookalike(l) for l in labels[:-1])),
    }
