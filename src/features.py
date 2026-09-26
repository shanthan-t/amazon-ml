"""Compact pairwise features for business-entity matching."""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from src.normalization import (
    normalize_business_address,
    normalize_business_name,
    normalize_country,
    normalized_tokens,
)


FEATURE_NAMES = (
    "name_exact",
    "name_token_jaccard",
    "name_char_similarity",
    "name_prefix6",
    "address_exact",
    "address_token_jaccard",
    "address_char_similarity",
    "numeric_jaccard",
    "country_equal",
    "name_length_ratio",
    "name_length_gap",
    "address_length_ratio",
    "address_length_gap",
    "s1_address_missing",
    "target_address_missing",
)

_DIGITS = re.compile(r"\d+")


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _ratio(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    return fuzz.ratio(left, right) / 100.0


def _length_features(left: str, right: str, scale: float) -> tuple[float, float]:
    left_len, right_len = len(left), len(right)
    maximum = max(left_len, right_len, 1)
    return min(left_len, right_len) / maximum, abs(left_len - right_len) / scale


def pair_features(
    s1_name: str,
    s1_address: str,
    s1_country: str,
    target_name: str,
    target_address: str,
    target_country: str,
) -> tuple[float, ...]:
    name1 = normalize_business_name(s1_name)
    name2 = normalize_business_name(target_name)
    address1 = normalize_business_address(s1_address)
    address2 = normalize_business_address(target_address)
    country1 = normalize_country(s1_country)
    country2 = normalize_country(target_country)
    return pair_features_normalized(name1, address1, country1, name2, address2, country2)


def pair_features_normalized(
    name1: str,
    address1: str,
    country1: str,
    name2: str,
    address2: str,
    country2: str,
    *,
    name_similarity: float | None = None,
) -> tuple[float, ...]:
    """Compute features from values already in the project's normalized form."""
    name_tokens1 = set(normalized_tokens(name1))
    name_tokens2 = set(normalized_tokens(name2))
    address_tokens1 = set(normalized_tokens(address1))
    address_tokens2 = set(normalized_tokens(address2))
    numbers1 = set(_DIGITS.findall(address1))
    numbers2 = set(_DIGITS.findall(address2))
    name_length_ratio, name_length_gap = _length_features(name1, name2, 100.0)
    address_length_ratio, address_length_gap = _length_features(
        address1, address2, 200.0
    )
    compact1 = "".join(name1.split())
    compact2 = "".join(name2.split())
    return (
        float(bool(name1) and name1 == name2),
        _jaccard(name_tokens1, name_tokens2),
        _ratio(name1, name2) if name_similarity is None else name_similarity,
        float(len(compact1) >= 6 and len(compact2) >= 6 and compact1[:6] == compact2[:6]),
        float(bool(address1) and address1 == address2),
        _jaccard(address_tokens1, address_tokens2),
        _ratio(address1, address2),
        _jaccard(numbers1, numbers2),
        float(bool(country1) and country1 == country2),
        name_length_ratio,
        name_length_gap,
        address_length_ratio,
        address_length_gap,
        float(not address1),
        float(not address2),
    )


def pair_features_exact_name(
    normalized_name: str,
    normalized_address: str,
    normalized_country: str,
    target_normalized_address: str,
    target_normalized_country: str,
) -> tuple[float, ...]:
    """Compute the regular feature vector when the exact-name route matched."""
    return pair_features_normalized(
        normalized_name,
        normalized_address,
        normalized_country,
        normalized_name,
        target_normalized_address,
        target_normalized_country,
        name_similarity=1.0,
    )
