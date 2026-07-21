"""Mechanical recovery of released TinyStories noun, verb, and adjective roles."""

from __future__ import annotations

import re

from apm.data.text.tinyworlds_p.contracts import Recipe
from apm.data.text.tinyworlds_p.normalization import normalize_ingredient_surface


_ROLE_NAMES = ("noun", "verb", "adjective")
_EXPLICIT_ROLE_PATTERN = re.compile(
    r"\b(?P<role>noun|verb|adjective)\b\s*(?:is\s+)?"
    r"[\"'\u2018\u201c](?P<word>[^\"'\u2018\u2019\u201c\u201d]{1,128})"
    r"[\"'\u2019\u201d]",
    re.IGNORECASE,
)


def recover_released_recipe(
    prompt_text: str,
    released_words: tuple[str, ...],
    released_features: tuple[str, ...] = (),
) -> Recipe | None:
    """Recover a recipe only when explicit prompt labels define unique roles."""
    if type(prompt_text) is not str or not prompt_text.strip():
        raise ValueError("released prompt must be nonempty text")
    if type(released_words) is not tuple or any(
        type(word) is not str or not word.strip() for word in released_words
    ):
        raise ValueError("released words must be a tuple of nonempty strings")
    if type(released_features) is not tuple or any(
        type(feature) is not str or not feature.strip()
        for feature in released_features
    ):
        raise ValueError("released features must be nonempty strings")
    normalized_words = tuple(
        normalize_ingredient_surface(word) for word in released_words
    )
    if len(normalized_words) != 3 or len(set(normalized_words)) != 3:
        return None
    matches = tuple(
        (
            match.group("role").casefold(),
            normalize_ingredient_surface(match.group("word")),
        )
        for match in _EXPLICIT_ROLE_PATTERN.finditer(prompt_text)
    )
    released_word_set = set(normalized_words)
    if any(word not in released_word_set for _, word in matches):
        return None
    by_role = {
        role: tuple(word for matched_role, word in matches if matched_role == role)
        for role in _ROLE_NAMES
    }
    if any(len(by_role[role]) != 1 for role in _ROLE_NAMES):
        return None
    assigned = tuple(by_role[role][0] for role in _ROLE_NAMES)
    if set(assigned) != released_word_set:
        return None
    return Recipe(
        noun=by_role["noun"][0],
        verb=by_role["verb"][0],
        adjective=by_role["adjective"][0],
        features=tuple(
            sorted(
                {
                    normalize_ingredient_surface(feature)
                    for feature in released_features
                }
            )
        ),
    )


__all__ = ["recover_released_recipe"]
