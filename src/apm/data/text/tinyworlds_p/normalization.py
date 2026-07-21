"""Conservative identity-only normalization for TinyWorlds-P source joins."""

from __future__ import annotations

from hashlib import sha256
import re
import unicodedata


_WHITESPACE = re.compile(r"\s+", flags=re.UNICODE)
_QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u2035": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2033": '"',
        "\u2036": '"',
    }
)


def normalize_story_identity(text: str) -> str:
    """Normalize text for exact identity without changing persisted training bytes."""
    if type(text) is not str:
        raise TypeError("story identity input must be text")
    compatibility_normalized = unicodedata.normalize("NFKC", text)
    straight_quotes = compatibility_normalized.translate(_QUOTE_TRANSLATION)
    return _WHITESPACE.sub(" ", straight_quotes.casefold()).strip()


def normalized_story_sha256(text: str) -> str:
    """Hash the UTF-8 bytes of the normalized story identity."""
    return sha256(normalize_story_identity(text).encode("utf-8")).hexdigest()


def normalized_story_bytes_sha256(raw_story: bytes) -> str:
    """Strictly decode raw corpus bytes and return their normalized identity hash."""
    if type(raw_story) is not bytes:
        raise TypeError("raw story must be bytes")
    return normalized_story_sha256(raw_story.decode("utf-8", errors="strict"))


def normalize_ingredient_surface(value: str) -> str:
    """Apply the v1 NFKC/case-fold/trim contract to one recipe ingredient."""
    if type(value) is not str:
        raise TypeError("ingredient surface must be text")
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    if not normalized:
        raise ValueError("ingredient surface must not normalize to empty text")
    return normalized


__all__ = [
    "normalize_ingredient_surface",
    "normalize_story_identity",
    "normalized_story_bytes_sha256",
    "normalized_story_sha256",
]
