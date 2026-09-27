"""Conservative Unicode-aware normalization for business text fields."""

import unicodedata


_ASCII_TO_SPACE = str.maketrans(
    {
        codepoint: " "
        for codepoint in range(128)
        if not chr(codepoint).isalnum() and not chr(codepoint).isspace()
    }
)


def _normalize_text(value: object, *, expand_ampersand: bool) -> str:
    if value is None:
        return ""

    text = unicodedata.normalize("NFKC", str(value)).casefold()
    if expand_ampersand:
        text = text.replace("&", " and ")
    text = text.translate(_ASCII_TO_SPACE)
    if text.isascii():
        return " ".join(text.split())

    normalized = []
    for char in text:
        if ord(char) < 128 or unicodedata.category(char)[0] in {"L", "M", "N"}:
            normalized.append(char)
        else:
            normalized.append(" ")
    return " ".join("".join(normalized).split())


def normalize_business_name(value: object) -> str:
    """Normalize compatibility forms, case, punctuation, and spacing.

    Legal suffixes and script are retained. Ampersands are represented as
    ``and``; other punctuation and symbols become token boundaries.
    """
    return _normalize_text(value, expand_ampersand=True)


def normalize_business_address(value: object) -> str:
    """Normalize address text while retaining every letter and number."""
    return _normalize_text(value, expand_ampersand=True)


def normalize_country(value: object) -> str:
    """Normalize country spelling and whitespace without a fixed country list."""
    return _normalize_text(value, expand_ampersand=False)


def normalized_tokens(value: str) -> tuple[str, ...]:
    """Split an already-normalized value into unique tokens in source order."""
    return tuple(dict.fromkeys(value.split()))
