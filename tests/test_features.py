"""
Unit tests for URL parsing and feature extraction (no dataset or trained model needed).

Run from the project root:   python -m unittest discover tests -v
"""
import unittest

from src.Feature_Pipeline import extract_features, feature_columns
from src.URL_Parser import parse_url


def feats(url):
    return extract_features(url)[1]


class ParserTests(unittest.TestCase):
    def test_scheme_and_trailing_slash_do_not_change_canonical_form(self):
        a = parse_url("http://Example.com/").canonical
        b = parse_url("example.com").canonical
        c = parse_url("https://EXAMPLE.com").canonical
        self.assertEqual(a, b)
        self.assertEqual(b, c)

    def test_had_scheme_is_recorded(self):
        self.assertTrue(parse_url("https://a.com").had_scheme)
        self.assertFalse(parse_url("a.com/x").had_scheme)

    def test_registered_domain_splitting(self):
        p = parse_url("http://login.secure.paypal.com.evil.xyz/a")
        self.assertEqual(p.registered_domain, "evil.xyz")
        self.assertEqual(p.subdomain, "login.secure.paypal.com")
        self.assertEqual(parse_url("www.bbc.co.uk").registered_domain, "bbc.co.uk")

    def test_invalid_inputs(self):
        for bad in (None, "", "   ", "http://", "not a url", "localhost", float("nan")):
            self.assertFalse(parse_url(bad).is_valid, msg=repr(bad))

    def test_zero_width_characters_are_counted_and_removed(self):
        p = parse_url("http://ex\u200bample.com")
        self.assertEqual(p.canonical, "example.com")
        self.assertEqual(p.hidden_chars, 1)


class FeatureTests(unittest.TestCase):
    def test_feature_count_is_stable(self):
        self.assertEqual(len(feature_columns()), len(feats("example.com")))

    def test_ip_hosts(self):
        self.assertEqual(feats("http://192.168.1.1/login")["is_ip_host"], 1)
        self.assertEqual(feats("http://0x7f000001/")["is_ip_obfuscated"], 1)
        self.assertEqual(feats("http://example.com")["is_ip_host"], 0)

    def test_punycode_and_non_ascii(self):
        f = feats("http://bücher.de")
        self.assertEqual(f["has_punycode"], 1)
        self.assertEqual(f["has_non_ascii_host"], 1)

    def test_brand_features(self):
        self.assertEqual(feats("paypal.com")["brand_in_subdomain"], 0)        # the real thing
        self.assertEqual(feats("paypal.com.evil.xyz")["brand_in_subdomain"], 1)
        self.assertEqual(feats("paypa1.com")["brand_lookalike_host"], 1)
        self.assertEqual(feats("www.google.com")["brand_lookalike_host"], 0)

    def test_userinfo_port_tld(self):
        f = feats("http://paypal.com@evil.tk:8080/")
        self.assertEqual(f["has_userinfo"], 1)
        self.assertEqual(f["nonstandard_port"], 1)
        self.assertEqual(f["tld_is_risky"], 1)

    def test_entropy_higher_for_random_strings(self):
        self.assertGreater(feats("a8f3k2x9q1z7.com/q9w8e7r6t5y4")["url_entropy"],
                           feats("aaaaaaaa.com/aaaa")["url_entropy"])

    def test_scheme_features_off_by_default(self):
        self.assertNotIn("is_https", feats("https://example.com"))


if __name__ == "__main__":
    unittest.main()
