"""
Network_Features.py - structural "how would a browser reach this?" features.

IMPORTANT: despite the file name, NOTHING here touches the network. The assignment says
the app runs fully offline and never fetches the page, so these are purely string-based
signals about the address mechanics (IP instead of a name, odd ports, '@' tricks,
link shorteners). No DNS, WHOIS, certificate or page-content features are used.
"""
from src import config

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "rebrand.ly",
    "cutt.ly", "shorturl.at", "tiny.cc", "rb.gy", "lnkd.in", "t.ly", "bl.ink", "adf.ly",
    "bit.do", "shorte.st",
}


def network_features(p, use_scheme=None):
    """p is a ParsedURL (valid). Returns a dict of numeric features."""
    if use_scheme is None:
        use_scheme = config.USE_SCHEME_FEATURES
    feats = {
        # raw IP instead of a name: no brand to verify, typical of throw-away servers
        "is_ip_host": int(p.is_ip),
        # hex / decimal / octal IP writing exists only to evade filters
        "is_ip_obfuscated": int(p.is_ip_obfuscated),
        "has_port": int(p.port is not None),
        "nonstandard_port": int(p.port is not None and p.port not in (80, 443)),
        # 'http://paypal.com@evil.com' - text before '@' is ignored by the browser
        "has_userinfo": int(bool(p.userinfo)),
        # shorteners hide the destination
        "is_shortener": int(p.registered_domain in SHORTENERS),
    }
    if use_scheme:
        # Off by default: the presence of a scheme can be an artefact of how a dataset was
        # collected rather than a real signal (see config.USE_SCHEME_FEATURES).
        feats["has_scheme"] = int(p.had_scheme)
        feats["is_https"] = int(p.scheme == "https")
    return feats
