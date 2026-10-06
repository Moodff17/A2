"""
URL_Parser.py - turn ANY raw record into one consistent, parsed form.

WHY THIS FILE EXISTS (Assignment 2 - "innovation in data processing")
---------------------------------------------------------------------
Our sources do not arrive in the same shape:
  * the Kaggle base data mixes full URLs ("http://a.com/x?y=1") with scheme-less
    ones ("a.com/x") and has inconsistent trailing slashes and casing;
  * a top-sites list such as Tranco gives bare domains only ("example.com");
  * PhishTank / OpenPhish give full URLs, often with percent-encoding or
    non-ASCII (internationalised) hostnames;
  * some strings contain stray whitespace, quotes, control or zero-width characters.
If these went straight into feature extraction, the model could learn the DATASET
(e.g. "no scheme => benign", "bare domain => benign") instead of phishing signals.

parse_url() therefore:
  1. cleans the text (decode bytes, strip whitespace/quotes/control/zero-width chars);
  2. records whether a scheme was present, then removes it from the canonical form;
  3. splits userinfo / host / port / path / query / fragment (RFC 3986 structure);
  4. converts internationalised hostnames to ASCII (punycode) but remembers the
     original had non-ASCII characters (a look-alike signal);
  5. splits the host into subdomain / domain / public suffix without any network
     lookup (small built-in table of two-part suffixes, e.g. co.uk, com.au);
  6. builds a canonical string used for de-duplication and for lexical features.

Nothing in this file touches the network - the URL is only ever treated as text.

Reference: Berners-Lee, Fielding & Masinter (2005) RFC 3986.
"""
import ipaddress
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_WS_CTRL_RE = re.compile(r"[\x00-\x1f\x7f\s]+")
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)

# Public suffixes made of two labels. A full public-suffix list would need a download,
# which we avoid so the tool stays fully offline; this covers the common cases.
_TWO_LEVEL_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "me.uk", "com.au", "net.au", "org.au",
    "edu.au", "gov.au", "co.nz", "org.nz", "co.jp", "ne.jp", "or.jp", "ac.jp",
    "com.br", "net.br", "org.br", "com.cn", "net.cn", "org.cn", "co.in", "net.in",
    "org.in", "com.mx", "com.ar", "co.za", "com.tr", "com.sg", "com.hk", "com.tw",
    "co.kr", "com.my", "com.ph", "com.vn", "co.id", "com.ua", "com.ng", "com.pk",
    "co.th", "com.co", "com.pe", "com.eg", "com.sa", "co.il",
}


@dataclass
class ParsedURL:
    """Everything later feature code needs, so no other file re-parses the string."""
    raw: str = ""
    is_valid: bool = False
    canonical: str = ""          # scheme-free, lower-case host, lone trailing '/' removed
    had_scheme: bool = False
    scheme: str = ""
    userinfo: str = ""
    host: str = ""               # as written (lower-case); may contain non-ASCII
    host_ascii: str = ""         # punycode form
    port: int = None
    path: str = ""
    query: str = ""
    fragment: str = ""
    subdomain: str = ""
    domain_label: str = ""       # e.g. "paypal" in www.paypal.com
    suffix: str = ""             # e.g. "com" or "co.uk"
    registered_domain: str = ""  # e.g. "paypal.com"
    is_ip: bool = False
    is_ip_obfuscated: bool = False
    hidden_chars: int = 0        # count of zero-width characters removed during cleaning


def clean_text(value):
    """Return a whitespace/control-free string; also returns number of zero-width chars removed."""
    if value is None:
        return "", 0
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value)
    if text.strip().lower() == "nan":
        return "", 0
    before = len(text)
    text = text.translate(_ZERO_WIDTH)
    hidden = before - len(text)
    text = _WS_CTRL_RE.sub("", text)
    return text.strip("\"'<>"), hidden


def looks_like_ip(host):
    """Return (is_ip, is_obfuscated). Obfuscated = hex / decimal-integer / octal-style IPs."""
    try:
        ipaddress.ip_address(host)
        return True, False
    except ValueError:
        pass
    if re.fullmatch(r"0x[0-9a-f]+", host) or re.fullmatch(r"\d{8,10}", host):
        return True, True
    parts = host.split(".")
    if len(parts) == 4 and all(re.fullmatch(r"(0x[0-9a-f]+|\d+)", p) for p in parts):
        return True, True
    return False, False


def _split_netloc(netloc):
    """'user:pw@host:8080' -> (userinfo, host, port). Handles [ipv6]:port."""
    userinfo = ""
    if "@" in netloc:
        userinfo, netloc = netloc.rsplit("@", 1)
    port, host = None, netloc
    if netloc.startswith("["):
        end = netloc.find("]")
        if end != -1:
            host = netloc[1:end]
            rest = netloc[end + 1:]
            if rest.startswith(":") and rest[1:].isdigit():
                port = int(rest[1:])
    elif ":" in netloc:
        head, _, tail = netloc.rpartition(":")
        if tail.isdigit():
            host, port = head, int(tail)
        elif tail == "":
            host = head
    if port is not None and port > 65535:
        port = None
    return userinfo, host.lower().rstrip("."), port


def _to_ascii_host(host):
    """Internationalised host -> punycode ('bücher.de' -> 'xn--bcher-kva.de')."""
    try:
        return host.encode("idna").decode("ascii")
    except (UnicodeError, ValueError):
        return host


def _split_domain(host_ascii):
    """Return (subdomain, domain_label, suffix, registered_domain) with no network lookup."""
    labels = [l for l in host_ascii.split(".") if l]
    if not labels:
        return "", "", "", ""
    if len(labels) >= 2 and ".".join(labels[-2:]) in _TWO_LEVEL_SUFFIXES:
        suffix = ".".join(labels[-2:])
        rest = labels[:-2]
    elif len(labels) >= 2:
        suffix = labels[-1]
        rest = labels[:-1]
    else:                                   # single-label host such as "localhost"
        return "", labels[0], "", labels[0]
    if not rest:                            # host IS a suffix (rare)
        return "", "", suffix, suffix
    domain_label = rest[-1]
    subdomain = ".".join(rest[:-1])
    return subdomain, domain_label, suffix, f"{domain_label}.{suffix}"


def parse_url(raw):
    """Parse one raw record into a ParsedURL. Never raises; is_valid=False on failure."""
    text, hidden = clean_text(raw)
    if not text:
        return ParsedURL(raw=str(raw))
    # A real address has no spaces inside it. Rejecting these stops free text ("not a url")
    # from being scored as if it were a URL.
    if isinstance(raw, str) and re.search(r"\s", raw.strip()):
        return ParsedURL(raw=str(raw))

    had_scheme = bool(_SCHEME_RE.match(text))
    scheme = ""
    if had_scheme:
        scheme = text.split("://", 1)[0].lower()
        to_parse = text
    else:
        to_parse = "http://" + text.lstrip("/")   # parse-only prefix; NOT kept in canonical

    try:
        parts = urlsplit(to_parse)
        userinfo, host, port = _split_netloc(parts.netloc)
    except ValueError:
        return ParsedURL(raw=str(raw))
    if not host:
        return ParsedURL(raw=str(raw))

    host_ascii = _to_ascii_host(host)
    is_ip, is_ip_obf = looks_like_ip(host_ascii)
    if not is_ip and "." not in host_ascii:      # needs at least one dot to be a real hostname
        return ParsedURL(raw=str(raw))
    if is_ip:
        subdomain, domain_label, suffix, registered = "", "", "", host_ascii
    else:
        subdomain, domain_label, suffix, registered = _split_domain(host_ascii)

    path = parts.path
    if path == "/" and not parts.query and not parts.fragment:
        path = ""                                # "a.com" and "a.com/" are the same record
    canonical = (
        (userinfo + "@" if userinfo else "")
        + host_ascii
        + (f":{port}" if port else "")
        + path
        + ("?" + parts.query if parts.query else "")
        + ("#" + parts.fragment if parts.fragment else "")
    )
    return ParsedURL(
        raw=str(raw), is_valid=True, canonical=canonical, had_scheme=had_scheme,
        scheme=scheme, userinfo=userinfo, host=host, host_ascii=host_ascii, port=port,
        path=path, query=parts.query, fragment=parts.fragment, subdomain=subdomain,
        domain_label=domain_label, suffix=suffix, registered_domain=registered,
        is_ip=is_ip, is_ip_obfuscated=is_ip_obf, hidden_chars=hidden,
    )
