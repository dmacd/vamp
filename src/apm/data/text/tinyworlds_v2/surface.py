"""Deterministic story normalization and surface-form measurements.

These checks deliberately do not use model-produced annotations.  They are
small, versionable measurements used both for source-cohort separation and for
the generated/reference quality comparison.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import re
import unicodedata


_LEXICAL_TOKEN_PATTERN = re.compile(
    r"[^\W_]+(?:['’-][^\W_]+)*",
    re.UNICODE,
)
_DIALOGUE_PATTERN = re.compile(
    r"[“\"][^“”\"\n]{1,240}[”\"]",
)
_FEATURE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "BadEnding": (
        re.compile(
            r"\b(?:sad|unhappy|alone|lost|broken|sorry|cried|crying|never returned|"
            r"could not|couldn't|did not come back|didn't come back)\b",
            re.IGNORECASE,
        ),
    ),
    "Conflict": (
        re.compile(
            r"\b(?:argu(?:e|ed|ing)|fight|fought|angry|mad at|would not|wouldn't|"
            r"stole|chased|trouble|problem|disagree(?:d|ment)?)\b",
            re.IGNORECASE,
        ),
    ),
    "Foreshadowing": (
        re.compile(
            r"\b(?:little did .{0,60} know|did not know|didn't know|would later|"
            r"soon (?:learn|find|discover)|strange feeling|warning|something .{0,30} wrong)\b",
            re.IGNORECASE,
        ),
    ),
    "MoralValue": (
        re.compile(
            r"\b(?:learned (?:a |the )?(?:lesson|importance)|lesson|moral|"
            r"from then on|it is important|it's important|always be|should always)\b",
            re.IGNORECASE,
        ),
    ),
    "Twist": (
        re.compile(
            r"\b(?:suddenly|to (?:his|her|their|everyone's) surprise|but then|"
            r"turned out|instead|unexpected(?:ly)?|all along)\b",
            re.IGNORECASE,
        ),
    ),
}
_CANONICAL_FEATURE_LABELS = ("Dialogue", *_FEATURE_PATTERNS)
_FEATURE_LABEL_BY_CASEFOLD = {
    unicodedata.normalize("NFC", label).casefold(): label
    for label in _CANONICAL_FEATURE_LABELS
}


def canonical_feature_labels(features: tuple[str, ...]) -> tuple[str, ...]:
    """Canonicalize presence labels while raw source metadata stays untouched."""
    if type(features) is not tuple or any(
        type(feature) is not str or not feature.strip() for feature in features
    ):
        raise ValueError("features must contain nonempty strings")
    by_key: dict[str, str] = {}
    for feature in features:
        normalized = unicodedata.normalize("NFC", feature).strip()
        key = normalized.casefold()
        by_key.setdefault(key, _FEATURE_LABEL_BY_CASEFOLD.get(key, normalized))
    return tuple(by_key[key] for key in sorted(by_key))


def normalized_story_text(story: str) -> str:
    """Return the comparison form used only for cross-cohort identity checks."""
    if type(story) is not str:
        raise TypeError("story must be a string")
    return " ".join(unicodedata.normalize("NFKC", story).casefold().split())


def normalized_story_sha256(story: str) -> str:
    """Hash a Unicode- and whitespace-normalized, case-folded story."""
    return sha256(normalized_story_text(story).encode("utf-8")).hexdigest()


def lexical_tokens(story: str) -> tuple[str, ...]:
    """Tokenize words without discarding numeric or alphanumeric forms."""
    if type(story) is not str:
        raise TypeError("story must be a string")
    return tuple(
        match.group(0).casefold() for match in _LEXICAL_TOKEN_PATTERN.finditer(story)
    )


def realized_feature_labels(
    story: str,
    requested_features: tuple[str, ...],
) -> tuple[str, ...]:
    """Detect conservatively realized released TinyStories narrative features.

    The detector is intentionally lexical and conservative.  It is applied to
    genuine and generated stories identically, so copied response annotations
    can never manufacture a positive realization.
    """
    requested_features = canonical_feature_labels(requested_features)
    detected: list[str] = []
    for feature in requested_features:
        if feature == "Dialogue":
            present = _DIALOGUE_PATTERN.search(story) is not None
        else:
            patterns = _FEATURE_PATTERNS.get(feature, ())
            present = any(pattern.search(story) is not None for pattern in patterns)
        if present:
            detected.append(feature)
    return tuple(sorted(detected))


def repeated_ngram_fraction(tokens: tuple[str, ...]) -> float:
    """Return repeated occurrences beyond first across contiguous 3--5-grams."""
    if type(tokens) is not tuple or any(type(token) is not str for token in tokens):
        raise TypeError("tokens must be a tuple of strings")
    total = 0
    repeated = 0
    for width in (3, 4, 5):
        ngrams = tuple(zip(*(tokens[offset:] for offset in range(width))))
        counts = Counter(ngrams)
        total += len(ngrams)
        repeated += sum(count - 1 for count in counts.values() if count > 1)
    return repeated / total if total else 0.0


def token_form_counts(tokens: tuple[str, ...]) -> tuple[int, int, int]:
    """Return digit-bearing, numeric-only, and identifier-like token counts.

    A mixed token must contain a letter and digit in the same hyphen/apostrophe
    segment.  Thus ``fox7`` and ``R2-D2`` are identifier-like, while ordinary
    ``3-year-old`` prose is merely digit-bearing.
    """
    if type(tokens) is not tuple or any(type(token) is not str for token in tokens):
        raise TypeError("tokens must be a tuple of strings")
    digit_bearing = sum(any(character.isdigit() for character in token) for token in tokens)
    numeric = sum(token.isnumeric() for token in tokens)
    mixed_alphanumeric = sum(
        any(
            any(character.isdigit() for character in segment)
            and any(character.isalpha() for character in segment)
            for segment in re.split(r"['’-]", token)
        )
        for token in tokens
    )
    return digit_bearing, numeric, mixed_alphanumeric


__all__ = [
    "canonical_feature_labels",
    "lexical_tokens",
    "normalized_story_sha256",
    "normalized_story_text",
    "realized_feature_labels",
    "repeated_ngram_fraction",
    "token_form_counts",
]
