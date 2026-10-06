"""
Lexical_Features.py - character-level features of the URL string.

Every feature answers "what does the TEXT of this address look like?", with no
lookups of any kind. Each feature below lists what it is meant to capture and the
published work that motivates it (full references: docs/feature_justification.md).

All features are computed on the CANONICAL string (scheme removed, see URL_Parser.py)
so that full URLs and bare domains are measured the same way.
"""
import math
from collections import Counter

# Words phishing pages commonly put in the address to look official or create urgency.
# Motivation: Garera et al. (2007); Sahingoz et al. (2019); Mamun et al. (2016).
SUSPICIOUS_KEYWORDS = [
    "login", "signin", "sign-in", "signon", "verify", "verification", "secure", "security",
    "account", "update", "confirm", "password", "passwd", "credential", "authenticate",
    "validate", "banking", "wallet", "webscr", "billing", "invoice", "payment", "suspend",
    "unlock", "recover", "support", "alert", "ebayisapi",
]


def shannon_entropy(text):
    """Shannon (1948) entropy in bits/char. Random-looking strings (generated domains,
    long tokens) score high; natural words score lower."""
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _longest_run(text, predicate):
    best = cur = 0
    for ch in text:
        cur = cur + 1 if predicate(ch) else 0
        best = max(best, cur)
    return best


def lexical_features(p):
    """p is a ParsedURL (valid). Returns a dict of numeric features."""
    s = p.canonical
    n = max(len(s), 1)
    low = s.lower()
    digits = sum(ch.isdigit() for ch in s)
    letters = sum(ch.isalpha() for ch in s)
    uppers = sum(ch.isupper() for ch in s)
    specials = sum(not ch.isalnum() for ch in s)
    host_tokens = [t for t in p.host_ascii.replace("-", ".").split(".") if t]
    path_segments = [seg for seg in p.path.split("/") if seg]

    return {
        # --- size: long URLs hide the real domain / carry encoded payloads (Ma et al. 2009)
        "url_length": len(s),
        "host_length": len(p.host_ascii),
        "path_length": len(p.path),
        "query_length": len(p.query),
        # --- character composition: phishing URLs are noisier than hand-typed ones
        "digit_count": digits,
        "digit_ratio": digits / n,
        "letter_ratio": letters / n,
        "uppercase_ratio": uppers / n,
        "special_char_ratio": specials / n,
        "longest_digit_run": _longest_run(s, str.isdigit),
        # --- punctuation counts: each is a known obfuscation device
        "hyphen_count": s.count("-"),       # 'secure-paypal-login' style compounds
        "dot_count": s.count("."),          # many dots = deep subdomain chains
        "underscore_count": s.count("_"),
        "at_count": s.count("@"),           # 'user@host' hides the true destination
        "question_count": s.count("?"),
        "equals_count": s.count("="),
        "ampersand_count": s.count("&"),
        "slash_count": s.count("/"),
        "percent_encoded_count": low.count("%"),   # encoded chars can mask keywords
        "double_slash_in_path": int("//" in p.path),  # open-redirect style trick
        # --- structure
        "path_depth": len(path_segments),
        "query_param_count": len(p.query.split("&")) if p.query else 0,
        "longest_host_token": max((len(t) for t in host_tokens), default=0),
        # --- randomness (Shannon 1948): algorithmically generated hosts/paths look random
        "url_entropy": shannon_entropy(s),
        "host_entropy": shannon_entropy(p.host_ascii),
        "path_entropy": shannon_entropy(p.path + p.query),
        # --- content words
        "suspicious_keyword_count": sum(1 for k in SUSPICIOUS_KEYWORDS if k in low),
        # --- hidden characters stripped by the cleaner (zero-width spaces are used to evade filters)
        "hidden_char_count": p.hidden_chars,
    }
