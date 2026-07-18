"""Versioned, split-disjoint deterministic TinyWorlds text templates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from apm.data.text.tinyworlds.schema import DataSplit
from apm.data.text.tinyworlds.world_generation import SymbolicWorld


TEMPLATE_REGISTRY_VERSION = "tinyworlds-templates-v1"


class TemplateKind(str, Enum):
    """The symbolic surface rendered by one template family."""

    FACT = "fact"
    RULE = "rule"
    QUERY = "query"
    PLOT = "plot"


@dataclass(frozen=True, slots=True)
class TemplateFamily:
    """One immutable split-specific prose frame for a symbolic target."""

    family_id: str
    split: DataSplit
    kind: TemplateKind
    target_id: str
    frame: str

    def __post_init__(self) -> None:
        if any(
            type(value) is not str or not value or value != value.strip()
            for value in (self.family_id, self.target_id, self.frame)
        ):
            raise ValueError("template identifiers and frames must be nonempty")
        if type(self.split) is not DataSplit:
            raise TypeError("template split must be a DataSplit")
        if type(self.kind) is not TemplateKind:
            raise TypeError("template kind must be a TemplateKind")
        if "{statement}" not in self.frame:
            raise ValueError("template frames must contain {statement}")

    def render(self, statement: str) -> str:
        """Render a nonempty symbolic statement into this prose frame."""
        if type(statement) is not str or not statement.strip():
            raise ValueError("template statements must contain visible text")
        return self.frame.format(statement=statement.strip())


@dataclass(frozen=True, slots=True)
class TinyWorldsTemplateRegistry:
    """Complete split-disjoint template registry for one symbolic world."""

    version: str
    families: tuple[TemplateFamily, ...]

    def __post_init__(self) -> None:
        if self.version != TEMPLATE_REGISTRY_VERSION:
            raise ValueError(
                f"template version must equal {TEMPLATE_REGISTRY_VERSION}"
            )
        if type(self.families) is not tuple or not self.families or any(
            type(family) is not TemplateFamily for family in self.families
        ):
            raise TypeError("template families must be a nonempty tuple")
        family_ids = tuple(family.family_id for family in self.families)
        if len(set(family_ids)) != len(family_ids):
            raise ValueError("template family IDs must be unique")
        target_kinds = tuple(
            dict.fromkeys((family.kind, family.target_id) for family in self.families)
        )
        for kind, target_id in target_kinds:
            matching = tuple(
                family
                for family in self.families
                if family.kind is kind and family.target_id == target_id
            )
            if tuple(family.split for family in matching) != tuple(DataSplit):
                raise ValueError(
                    "every template target requires train, validation, and test families"
                )
        split_ids = tuple(
            {
                family.family_id
                for family in self.families
                if family.split is split
            }
            for split in DataSplit
        )
        if any(
            split_ids[left] & split_ids[right]
            for left, right in ((0, 1), (0, 2), (1, 2))
        ):
            raise ValueError("template family IDs must be disjoint across splits")

    def family(
        self,
        kind: TemplateKind,
        target_id: str,
        split: DataSplit,
    ) -> TemplateFamily:
        """Resolve the sole family for a symbolic target and split."""
        matches = tuple(
            family
            for family in self.families
            if family.kind is kind
            and family.target_id == target_id
            and family.split is split
        )
        if len(matches) != 1:
            raise KeyError(
                f"no unique {kind.value} template for {target_id!r} in {split.value}"
            )
        return matches[0]


_SPLIT_FRAMES = {
    DataSplit.TRAIN: (
        "Long ago, {statement}",
        "The little village remembered that {statement}",
        "A patient storyteller explained: {statement}",
        "In the next part of the tale, {statement}",
    ),
    DataSplit.VALIDATION: (
        "On a gentle morning, {statement}",
        "The village notebook carefully said that {statement}",
        "A quiet witness then recalled: {statement}",
        "The practice question asked: {statement}",
    ),
    DataSplit.TEST: (
        "Beneath the evening lanterns, {statement}",
        "The final storybook plainly recorded that {statement}",
        "A new visitor later discovered: {statement}",
        "The last question wondered: {statement}",
    ),
}


def build_template_registry(world: SymbolicWorld) -> TinyWorldsTemplateRegistry:
    """Build complete predicate, rule, query, and plot families for a world."""
    if type(world) is not SymbolicWorld:
        raise TypeError("world must be a SymbolicWorld")
    targets = (
        *(
            (TemplateKind.FACT, str(predicate.predicate_id))
            for predicate in world.registry.predicates
        ),
        *((TemplateKind.RULE, str(rule.rule_id)) for rule in world.rules),
        *(
            (TemplateKind.QUERY, query_kind)
            for query_kind in (
                "direct",
                "ancestor_plus_child",
                "new_instance",
                "one_hop",
                "two_hop",
                "revision_sensitive",
                "cross_branch",
                "open_book",
            )
        ),
        *((TemplateKind.PLOT, task.kind.value) for task in world.tasks),
        (TemplateKind.PLOT, "root_validation"),
    )
    unique_targets = tuple(dict.fromkeys(targets))
    families = tuple(
        TemplateFamily(
            family_id=(
                f"template:{TEMPLATE_REGISTRY_VERSION}:{split.value}:"
                f"{kind.value}:{target_id}"
            ),
            split=split,
            kind=kind,
            target_id=target_id,
            frame=_SPLIT_FRAMES[split][list(TemplateKind).index(kind)],
        )
        for kind, target_id in unique_targets
        for split in DataSplit
    )
    return TinyWorldsTemplateRegistry(TEMPLATE_REGISTRY_VERSION, families)


__all__ = [
    "TEMPLATE_REGISTRY_VERSION",
    "TemplateFamily",
    "TemplateKind",
    "TinyWorldsTemplateRegistry",
    "build_template_registry",
]
