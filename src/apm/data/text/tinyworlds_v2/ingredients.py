"""Mechanical ingredient-role parsing for released TinyStories prompts."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


_INGREDIENT_ROLES = ("noun", "verb", "adjective")
_EXPLICIT_ROLE_PATTERN = re.compile(
    r"\b(?P<role>noun|verb|adjective)\b\s*(?:is\s+)?"
    r"[\"'\u2018\u201c](?P<word>[^\"'\u2018\u2019\u201c\u201d]{1,128})"
    r"[\"'\u2019\u201d]",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class IngredientRoles:
    """Three released words mechanically linked to explicit prompt labels."""

    noun: str
    verb: str
    adjective: str

    def __post_init__(self) -> None:
        values = (self.noun, self.verb, self.adjective)
        if any(type(value) is not str or not value.strip() for value in values):
            raise ValueError("ingredient roles must contain nonempty strings")
        if len({_normalize_ingredient(value) for value in values}) != 3:
            raise ValueError("ingredient role words must be unique")


def mechanically_classify_ingredient_roles(
    prompt_text: str,
    released_words: tuple[str, ...],
) -> IngredientRoles | None:
    """Use explicit released prompt labels, returning ``None`` when ambiguous."""
    if type(prompt_text) is not str or not prompt_text.strip():
        raise ValueError("released prompt must be a nonempty string")
    if type(released_words) is not tuple or any(
        type(word) is not str or not word.strip() for word in released_words
    ):
        raise ValueError("released_words must contain nonempty strings")
    normalized_words = tuple(_normalize_ingredient(word) for word in released_words)
    if len(normalized_words) != 3 or len(set(normalized_words)) != 3:
        return None
    released_by_normalized = dict(zip(normalized_words, released_words, strict=True))
    matches = tuple(
        (match.group("role").casefold(), _normalize_ingredient(match.group("word")))
        for match in _EXPLICIT_ROLE_PATTERN.finditer(prompt_text)
    )
    if any(word not in released_by_normalized for _, word in matches):
        return None
    words_by_role = {
        role: tuple(word for matched_role, word in matches if matched_role == role)
        for role in _INGREDIENT_ROLES
    }
    if any(len(words_by_role[role]) != 1 for role in _INGREDIENT_ROLES):
        return None
    assigned = tuple(words_by_role[role][0] for role in _INGREDIENT_ROLES)
    if set(assigned) != set(normalized_words):
        return None
    return IngredientRoles(
        noun=released_by_normalized[words_by_role["noun"][0]],
        verb=released_by_normalized[words_by_role["verb"][0]],
        adjective=released_by_normalized[words_by_role["adjective"][0]],
    )


def _normalize_ingredient(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip().casefold()


__all__ = ["IngredientRoles", "mechanically_classify_ingredient_roles"]
