# Feature justification (for the report: Data Processing / Feature research)

Each feature is kept because of what it is meant to capture, not because it raised a score.
All features use the URL **string only** (no DNS, WHOIS, or page fetching) so the app stays offline.

| Group (file) | Features | What it is meant to capture | Source (verify before citing) |
|---|---|---|---|
| Size (Lexical_Features) | url_length, host_length, path_length, query_length | Long addresses hide the real domain or carry payloads/tracking tokens | Ma et al. (2009); Sahoo et al. (2017) |
| Composition (Lexical) | digit_count/ratio, letter_ratio, uppercase_ratio, special_char_ratio, longest_digit_run | Auto-generated and obfuscated URLs are noisier than hand-written ones | Mamun et al. (2016); Sahingoz et al. (2019) |
| Punctuation (Lexical) | hyphen, dot, underscore, @, ?, =, &, / counts; percent_encoded_count; double_slash_in_path | Each is a known obfuscation or redirection device | Sahingoz et al. (2019); Garera et al. (2007) |
| Structure (Lexical) | path_depth, query_param_count, longest_host_token | Deep paths / many parameters are typical of kit-generated phishing pages | Mamun et al. (2016) |
| Randomness (Lexical) | url_entropy, host_entropy, path_entropy | Algorithmically generated strings look random (high Shannon entropy) | Shannon (1948) |
| Keywords (Lexical) | suspicious_keyword_count | Urgency / credential words (login, verify, secure...) | Garera et al. (2007); Sahingoz et al. (2019) |
| Hidden chars (Lexical) | hidden_char_count | Zero-width characters are used to evade string filters | (our own observation - state it as such) |
| Host depth (Extract_Domains) | subdomain_count, host_label_count | `paypal.com.secure.evil.xyz` hides the real domain behind extra labels | Garera et al. (2007) |
| Domain label (Extract_Domains) | domain_label_length, domain_hyphen_count, domain_digit_count, mixed_alnum_label_count | Look-alike and throw-away domains mix digits/hyphens into the name | Sahingoz et al. (2019) |
| TLD (Extract_Domains) | tld_length, tld_is_risky, tld_is_common | Some TLDs are abused far more than others. The list is **static** (not learned from our labels) to avoid leakage | Spamhaus "most abused TLDs"; Interisle Consulting Group phishing landscape reports |
| Look-alikes (Extract_Domains) | has_punycode, has_non_ascii_host | Internationalised look-alike characters enable homograph attacks | Gabrilovich & Gontmakher (2002); Costello (2003) RFC 3492 |
| Brand misuse (Extract_Domains) | brand_in_subdomain, brand_in_path, brand_lookalike_host | A famous brand where it does not belong, or a misspelt copy of it | Garera et al. (2007) |
| Address mechanics (Network_Features) | is_ip_host, is_ip_obfuscated | A raw IP means no brand/domain to verify; hex/decimal IPs exist only to evade filters | Ma et al. (2009); Sahoo et al. (2017) |
| Address mechanics (Network_Features) | has_port, nonstandard_port, has_userinfo, is_shortener | Unusual ports, `user@host` tricks and shorteners hide the true destination | Sahoo et al. (2017) |

Excluded on purpose: DNS/WHOIS/domain-age/page-content features (need the network); scheme features
(`has_scheme`, `is_https`) are off by default because the dataset's scheme presence may be a collection
artefact (see `USE_SCHEME_FEATURES` in `src/config.py` and the table printed by `python Main.py explore`).

## Candidate references (check every detail before putting into the Harvard bibliography)
- Berners-Lee, T., Fielding, R. and Masinter, L. (2005) *RFC 3986: Uniform Resource Identifier (URI): Generic Syntax*.
- Chen, T. and Guestrin, C. (2016) 'XGBoost: a scalable tree boosting system', *KDD 2016*.
- Costello, A. (2003) *RFC 3492: Punycode*.
- Gabrilovich, E. and Gontmakher, A. (2002) 'The homograph attack', *Communications of the ACM*, 45(2).
- Garera, S., Provos, N., Chew, M. and Rubin, A. (2007) 'A framework for detection and measurement of phishing attacks', *WORM 2007*.
- Ke, G. et al. (2017) 'LightGBM: a highly efficient gradient boosting decision tree', *NeurIPS 2017*.
- Le Pochat, V. et al. (2019) 'Tranco: a research-oriented top sites ranking hardened against manipulation', *NDSS 2019*.
- Ma, J., Saul, L., Savage, S. and Voelker, G. (2009) 'Beyond blacklists: learning to detect malicious web sites from suspicious URLs', *KDD 2009*.
- Mamun, M. et al. (2016) 'Detecting malicious URLs using lexical analysis', *NSS 2016*.
- Sahingoz, O. et al. (2019) 'Machine learning based phishing detection from URLs', *Expert Systems with Applications*, 117.
- Sahoo, D., Liu, C. and Hoi, S. (2017) 'Malicious URL detection using machine learning: a survey', arXiv:1701.07179.
- Shannon, C. (1948) 'A mathematical theory of communication', *Bell System Technical Journal*, 27.
- Dataset: Siddhartha, M. (2021) *Malicious URLs dataset*, Kaggle (sid321axn/malicious-urls-dataset) - confirm author/year on the Kaggle page.
