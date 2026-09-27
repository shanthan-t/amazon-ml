"""V2 shared utilities: blocking keys + feature computation."""

from __future__ import annotations

import re
from rapidfuzz import fuzz
from .normalization import (
    normalize_business_address,
    normalize_business_name,
    normalize_country,
)

# ── Feature schema ────────────────────────────────────────────────────
V2_FEATURE_NAMES = (
    "name_exact",
    "name_token_jaccard",
    "name_fuzz_ratio",
    "name_prefix6",
    "name_prefix4",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_common_tokens",
    "name_common_token_ratio",
    "name_first_word_match",
    "address_exact",
    "address_token_jaccard",
    "address_fuzz_ratio",
    "address_token_sort_ratio",
    "numeric_jaccard",
    "country_equal",
    "name_length_ratio",
    "name_length_gap",
    "address_length_ratio",
    "address_length_gap",
    "s1_address_missing",
    "target_address_missing",
    # V4 new features
    "is_india",
    "target_is_s2",
    "name_is_short",
)

_DIGITS_RE = re.compile(r"\d+")

# Words too common to be useful as individual blocking keys.
_BLOCK_STOPWORDS = frozenset({
    "the", "and", "of", "for", "in", "at", "to", "a", "an", "by", "on",
    "inc", "llc", "ltd", "corp", "co", "pvt", "sa", "gmbh",
    "limited", "incorporated", "corporation", "company", "private",
    "services", "service", "solutions", "enterprises", "enterprise",
    "group", "international", "industries", "holdings", "consulting",
    "associates", "partners", "technologies", "technology",
})

MAX_POSTINGS_PER_KEY = 500


# ── Blocking key generation ──────────────────────────────────────────

def generate_blocking_keys(
    norm_name: str,
    norm_country: str,
) -> list[str]:
    """Generate multiple blocking keys from already-normalized values.

    Routes:
      E|  — exact normalized name (highest precision)
      P6| — 6-char prefix of compact name
      P4| — 4-char prefix of compact name (broader)
      ST| — sorted pairs of significant tokens (word-reorder safe)
      W|  — individual significant tokens (length ≥ 5)
    """
    if not norm_name or not norm_country:
        return []

    keys: list[str] = []
    compact = "".join(norm_name.split())
    tokens = sorted(set(norm_name.split()))

    # Route E: exact name
    keys.append(f"E|{norm_country}|{norm_name}")

    # Route P6: 6-char prefix
    if len(compact) >= 6:
        keys.append(f"P6|{norm_country}|{compact[:6]}")

    # Route P4: 4-char prefix
    if len(compact) >= 4:
        keys.append(f"P4|{norm_country}|{compact[:4]}")

    # Route ST: sorted significant-token pairs
    sig = [t for t in tokens if len(t) >= 3 and t not in _BLOCK_STOPWORDS]
    for i in range(len(sig)):
        for j in range(i + 1, min(i + 3, len(sig))):
            keys.append(f"ST|{norm_country}|{sig[i]}_{sig[j]}")

    # Route W: individual significant words (≥5 chars, not stopword)
    for token in tokens:
        if len(token) >= 5 and token not in _BLOCK_STOPWORDS:
            keys.append(f"W|{norm_country}|{token}")

    return keys


def generate_numeric_address_keys(norm_address: str, norm_country: str) -> list[str]:
    """Generate country-aware keys for address numbers with at least 3 digits."""
    if not norm_address or not norm_country:
        return []
    return [
        f"AN|{norm_country}|{number}"
        for number in sorted(set(_DIGITS_RE.findall(norm_address)))
        if len(number) >= 3
    ]


# ── Feature computation ──────────────────────────────────────────────

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def compute_pair_features(
    n1: str, a1: str, c1: str,
    n2: str, a2: str, c2: str,
    target_source: int = 0,
) -> tuple[float, ...]:
    """Compute 26-feature vector from already-normalized values.

    ``target_source`` is 2 for Source-2, 3 for Source-3, or 0 when unknown.
    """
    n_tok1 = set(n1.split()) if n1 else set()
    n_tok2 = set(n2.split()) if n2 else set()
    a_tok1 = set(a1.split()) if a1 else set()
    a_tok2 = set(a2.split()) if a2 else set()
    compact1 = "".join(n1.split())
    compact2 = "".join(n2.split())
    nums1 = set(_DIGITS_RE.findall(a1))
    nums2 = set(_DIGITS_RE.findall(a2))

    common = n_tok1 & n_tok2
    total = n_tok1 | n_tok2
    first1 = n1.split()[0] if n1 else ""
    first2 = n2.split()[0] if n2 else ""
    max_n = max(len(n1), len(n2), 1)
    max_a = max(len(a1), len(a2), 1)
    shorter_name_len = min(len(compact1), len(compact2))

    return (
        # Name features (11)
        float(bool(n1) and n1 == n2),                                      # name_exact
        _jaccard(n_tok1, n_tok2),                                          # name_token_jaccard
        fuzz.ratio(n1, n2) / 100.0,                                       # name_fuzz_ratio
        float(len(compact1) >= 6 and len(compact2) >= 6
              and compact1[:6] == compact2[:6]),                            # name_prefix6
        float(len(compact1) >= 4 and len(compact2) >= 4
              and compact1[:4] == compact2[:4]),                            # name_prefix4
        fuzz.token_sort_ratio(n1, n2) / 100.0,                            # name_token_sort_ratio
        fuzz.token_set_ratio(n1, n2) / 100.0,                             # name_token_set_ratio
        fuzz.partial_ratio(n1, n2) / 100.0,                               # name_partial_ratio
        float(len(common)),                                                # name_common_tokens
        len(common) / len(total) if total else 1.0,                        # name_common_token_ratio
        float(bool(first1) and first1 == first2),                          # name_first_word_match
        # Address features (5)
        float(bool(a1) and a1 == a2),                                      # address_exact
        _jaccard(a_tok1, a_tok2),                                          # address_token_jaccard
        fuzz.ratio(a1, a2) / 100.0,                                       # address_fuzz_ratio
        fuzz.token_sort_ratio(a1, a2) / 100.0,                            # address_token_sort_ratio
        _jaccard(nums1, nums2),                                            # numeric_jaccard
        # Structural features (7)
        float(bool(c1) and c1 == c2),                                      # country_equal
        min(len(n1), len(n2)) / max_n,                                     # name_length_ratio
        abs(len(n1) - len(n2)) / 100.0,                                    # name_length_gap
        min(len(a1), len(a2)) / max_a,                                     # address_length_ratio
        abs(len(a1) - len(a2)) / 200.0,                                    # address_length_gap
        float(not a1),                                                      # s1_address_missing
        float(not a2),                                                      # target_address_missing
        # V4 contextual features (3)
        float("india" in c1),                                              # is_india
        float(target_source == 2),                                         # target_is_s2
        float(shorter_name_len <= 5),                                      # name_is_short
    )


def compute_pair_features_raw(
    s1_name: str, s1_addr: str, s1_country: str,
    t_name: str, t_addr: str, t_country: str,
    target_source: int = 0,
) -> tuple[float, ...]:
    """Compute features from raw (un-normalized) values."""
    return compute_pair_features(
        normalize_business_name(s1_name),
        normalize_business_address(s1_addr),
        normalize_country(s1_country),
        normalize_business_name(t_name),
        normalize_business_address(t_addr),
        normalize_country(t_country),
        target_source=target_source,
    )
