"""Generate the deterministic TinyWorlds-P validation-sample appendix."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence, cast


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = REPOSITORY_ROOT / "data" / "tinyworlds-p-archive" / "v1"
OUTPUT_PATH = REPOSITORY_ROOT / "docs" / "TW-P_ARCHIVE_VALIDATION_SAMPLES.md"
WORLD_LABELS = ("A", "B", "C", "D", "E")
CONTROL_ARMS = ("same noun row", "same verb column")


@dataclass(frozen=True, slots=True)
class GridSpec:
    """Identify one immutable archive-only calibration grid."""

    title: str
    dimensions: str
    partition_sha256: str

    @property
    def root(self) -> Path:
        return ARCHIVE_ROOT / self.partition_sha256


@dataclass(frozen=True, slots=True)
class IndexRecord:
    """Retain the source and shard coordinates needed for one sample."""

    content_sha256: str
    normalized_story_sha256: str
    record_id: str
    source: str
    source_index: int
    source_member: str
    story_sha256: str
    text_bytes: int
    text_offset: int
    text_shard: int
    token_count: int


@dataclass(frozen=True, slots=True)
class IndexScan:
    """Retain streaming index statistics and one canonical record per length."""

    occurrence_count: int
    active_token_count: int
    median_token_count: int
    first_record_by_token_count: tuple[tuple[int, IndexRecord], ...]


@dataclass(frozen=True, slots=True)
class Recipe:
    """Describe the mechanically recovered recipe behind one story."""

    adjective: str
    features: tuple[str, ...]
    noun: str
    verb: str
    noun_bucket: int
    verb_bucket: int


@dataclass(frozen=True, slots=True)
class Sample:
    """Bind one exact story to its recipe, provenance, and optional control arm."""

    index: IndexRecord
    recipe: Recipe
    text: str
    control_arm: str | None


@dataclass(frozen=True, slots=True)
class Condition:
    """Summarize one validation index and its selected samples."""

    label: str
    occurrence_count: int
    active_token_count: int
    median_token_count: int
    samples: tuple[Sample, ...]


@dataclass(frozen=True, slots=True)
class WorldCell:
    """Describe one selected noun-bucket by verb-bucket cell."""

    label: str
    noun_bucket: int
    verb_bucket: int
    group_count: int
    active_token_count: int


@dataclass(frozen=True, slots=True)
class WordBucket:
    """Retain the words and mass assigned to one physical bucket."""

    namespace: str
    index: int
    token_mass: int
    words: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GridEvidence:
    """Contain every sample and topology fact rendered for one grid."""

    spec: GridSpec
    cells: tuple[WorldCell, ...]
    buckets: tuple[WordBucket, ...]
    base: Condition
    worlds: tuple[Condition, ...]
    controls: tuple[Condition, ...]


GRID_SPECS = (
    GridSpec(
        title="Initial 8×8 calibration",
        dimensions="8×8",
        partition_sha256=(
            "beb9e1e38efdf0447b9421b072c4053fdb7b6156c4814edefa170ec40072f084"
        ),
    ),
    GridSpec(
        title="Fresh 6×6 fallback calibration",
        dimensions="6×6",
        partition_sha256=(
            "7bf90c70ca7207d8b0fdd7896eed7a2ae019bbcbd74126cfcc2115ae0759b4fb"
        ),
    ),
)


def _json_object(payload: str, label: str) -> dict[str, object]:
    value = json.loads(payload)
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _required_string(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)
    if type(value) is not str:
        raise ValueError(f"{key} must be a string")
    return value


def _required_integer(record: Mapping[str, object], key: str) -> int:
    value = record.get(key)
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer")
    return value


def _index_record(line: str, path: Path) -> IndexRecord:
    record = _json_object(line, str(path))
    return IndexRecord(
        content_sha256=_required_string(record, "content_sha256"),
        normalized_story_sha256=_required_string(
            record, "normalized_story_sha256"
        ),
        record_id=_required_string(record, "record_id"),
        source=_required_string(record, "source"),
        source_index=_required_integer(record, "source_index"),
        source_member=_required_string(record, "source_member"),
        story_sha256=_required_string(record, "story_sha256"),
        text_bytes=_required_integer(record, "text_bytes"),
        text_offset=_required_integer(record, "text_offset"),
        text_shard=_required_integer(record, "text_shard"),
        token_count=_required_integer(record, "token_count"),
    )


def _scan_index(path: Path) -> IndexScan:
    counts_by_token: dict[int, int] = {}
    first_record_by_token: dict[int, IndexRecord] = {}
    occurrence_count = 0
    active_token_count = 0
    prior_order_key: tuple[str, str] | None = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = _index_record(line, path)
            order_key = (record.normalized_story_sha256, record.record_id)
            if prior_order_key is not None and order_key < prior_order_key:
                raise ValueError(f"validation index is not canonically ordered: {path}")
            prior_order_key = order_key
            occurrence_count += 1
            active_token_count += record.token_count
            counts_by_token[record.token_count] = (
                counts_by_token.get(record.token_count, 0) + 1
            )
            first_record_by_token.setdefault(record.token_count, record)
    if not occurrence_count:
        raise ValueError(f"validation index is empty: {path}")
    median_rank = (occurrence_count - 1) // 2
    cumulative = 0
    median_token_count = -1
    for token_count, count in sorted(counts_by_token.items()):
        cumulative += count
        if cumulative > median_rank:
            median_token_count = token_count
            break
    if median_token_count < 0:
        raise AssertionError("lower median was not found")
    return IndexScan(
        occurrence_count=occurrence_count,
        active_token_count=active_token_count,
        median_token_count=median_token_count,
        first_record_by_token_count=tuple(sorted(first_record_by_token.items())),
    )


def _find_assignment(path: Path, normalized_story_sha256: str) -> dict[str, object]:
    """Find a group in the normalized-story-sorted assignment JSONL by seeking."""
    with path.open("rb") as handle:
        low, high = 0, path.stat().st_size
        while low < high:
            middle = (low + high) // 2
            handle.seek(max(0, middle - 1))
            if middle:
                handle.readline()
            line_start = handle.tell()
            if line_start >= high:
                high = middle
                continue
            line = handle.readline()
            if not line:
                high = middle
                continue
            assignment = _json_object(line.decode("utf-8"), str(path))
            found = _required_string(assignment, "normalized_story_sha256")
            if found < normalized_story_sha256:
                low = handle.tell()
            elif found > normalized_story_sha256:
                # Forward alignment may place line_start beyond middle. Keeping
                # middle retains the partial preceding line and guarantees that
                # a variable-length search interval shrinks on every iteration.
                high = middle
            else:
                return assignment
    raise KeyError(f"assignment not found: {normalized_story_sha256}")


def _recipe(assignment: Mapping[str, object]) -> Recipe:
    value = assignment.get("recipe")
    if type(value) is not dict:
        raise ValueError("assignment recipe must be an object")
    recipe = cast(dict[str, object], value)
    features_value = recipe.get("features")
    if type(features_value) is not list or any(
        type(feature) is not str for feature in features_value
    ):
        raise ValueError("recipe features must be strings")
    return Recipe(
        adjective=_required_string(recipe, "adjective"),
        features=tuple(cast(list[str], features_value)),
        noun=_required_string(recipe, "noun"),
        verb=_required_string(recipe, "verb"),
        noun_bucket=_required_integer(assignment, "noun_bucket"),
        verb_bucket=_required_integer(assignment, "verb_bucket"),
    )


def _read_story(root: Path, index: IndexRecord) -> str:
    shard_path = root / "shards" / f"text-{index.text_shard:06d}.bin"
    with shard_path.open("rb") as handle:
        handle.seek(index.text_offset)
        story_bytes = handle.read(index.text_bytes)
    if len(story_bytes) != index.text_bytes:
        raise ValueError(f"short story read for {index.record_id}")
    if sha256(story_bytes).hexdigest() != index.story_sha256:
        raise ValueError(f"story hash mismatch for {index.record_id}")
    return story_bytes.decode("utf-8")


def _sample(
    root: Path,
    index: IndexRecord,
    assignment: Mapping[str, object],
    control_arm: str | None,
) -> Sample:
    return Sample(
        index=index,
        recipe=_recipe(assignment),
        text=_read_story(root, index),
        control_arm=control_arm,
    )


def _ordinary_condition(root: Path, index_name: str, label: str) -> Condition:
    scan = _scan_index(root / "indexes" / index_name)
    chosen = dict(scan.first_record_by_token_count)[scan.median_token_count]
    assignment = _find_assignment(
        root / "assignments.jsonl", chosen.normalized_story_sha256
    )
    return Condition(
        label=label,
        occurrence_count=scan.occurrence_count,
        active_token_count=scan.active_token_count,
        median_token_count=scan.median_token_count,
        samples=(_sample(root, chosen, assignment, None),),
    )


def _control_arm(recipe: Recipe, cell: WorldCell) -> str:
    if recipe.noun_bucket == cell.noun_bucket and recipe.verb_bucket != cell.verb_bucket:
        return "same noun row"
    if recipe.verb_bucket == cell.verb_bucket and recipe.noun_bucket != cell.noun_bucket:
        return "same verb column"
    raise ValueError(
        f"control recipe N{recipe.noun_bucket}×V{recipe.verb_bucket} is not "
        f"an arm of world {cell.label}"
    )


def _control_condition(root: Path, cell: WorldCell) -> Condition:
    label = f"control/{cell.label}/validation"
    index_path = root / "indexes" / f"control-{cell.label}-validation.jsonl"
    scan = _scan_index(index_path)
    assignments_path = root / "assignments.jsonl"

    selected: dict[str, Sample] = {}
    maximum_distance = max(
        abs(token_count - scan.median_token_count)
        for token_count, _record in scan.first_record_by_token_count
    )
    for distance in range(maximum_distance + 1):
        target_counts = {
            scan.median_token_count - distance,
            scan.median_token_count + distance,
        }
        with index_path.open(encoding="utf-8") as handle:
            for line in handle:
                index = _index_record(line, index_path)
                if index.token_count not in target_counts:
                    continue
                assignment = _find_assignment(
                    assignments_path, index.normalized_story_sha256
                )
                arm = _control_arm(_recipe(assignment), cell)
                selected.setdefault(arm, _sample(root, index, assignment, arm))
                if all(arm_name in selected for arm_name in CONTROL_ARMS):
                    break
        if all(arm_name in selected for arm_name in CONTROL_ARMS):
            break
    if not all(arm in selected for arm in CONTROL_ARMS):
        raise ValueError(f"{label} does not cover both control arms")

    return Condition(
        label=label,
        occurrence_count=scan.occurrence_count,
        active_token_count=scan.active_token_count,
        median_token_count=scan.median_token_count,
        samples=tuple(selected[arm] for arm in CONTROL_ARMS),
    )


def _world_cells(root: Path) -> tuple[WorldCell, ...]:
    topology = _json_object((root / "topology.json").read_text(), "topology")
    cells_value = topology.get("cells")
    if type(cells_value) is not list:
        raise ValueError("topology cells must be a list")
    cells = tuple(
        WorldCell(
            label=_required_string(cast(dict[str, object], value), "label"),
            noun_bucket=_required_integer(
                cast(dict[str, object], value), "noun_bucket"
            ),
            verb_bucket=_required_integer(
                cast(dict[str, object], value), "verb_bucket"
            ),
            group_count=_required_integer(
                cast(dict[str, object], value), "group_count"
            ),
            active_token_count=_required_integer(
                cast(dict[str, object], value), "active_token_count"
            ),
        )
        for value in cells_value
        if type(value) is dict
    )
    if tuple(cell.label for cell in cells) != WORLD_LABELS:
        raise ValueError("topology cells must be ordered A through E")
    return cells


def _word_buckets(root: Path) -> tuple[WordBucket, ...]:
    payload = _json_object((root / "buckets.json").read_text(), "buckets")
    buckets_value = payload.get("buckets")
    if type(buckets_value) is not list:
        raise ValueError("buckets must be a list")

    def parse_bucket(value: object) -> WordBucket:
        if type(value) is not dict:
            raise ValueError("bucket entry must be an object")
        record = cast(dict[str, object], value)
        words_value = record.get("words")
        if type(words_value) is not list or any(
            type(word) is not str for word in words_value
        ):
            raise ValueError("bucket words must be strings")
        return WordBucket(
            namespace=_required_string(record, "namespace"),
            index=_required_integer(record, "index"),
            token_mass=_required_integer(record, "token_mass"),
            words=tuple(cast(list[str], words_value)),
        )

    return tuple(parse_bucket(value) for value in buckets_value)


def _collect_grid(spec: GridSpec) -> GridEvidence:
    started = perf_counter()
    cells = _world_cells(spec.root)
    base = _ordinary_condition(
        spec.root, "base-validation.jsonl", "base/validation"
    )
    print(
        f"  [{spec.dimensions} 1/11] base/validation; grid ETA "
        f"{10 * (perf_counter() - started):.0f}s",
        flush=True,
    )
    worlds: list[Condition] = []
    for ordinal, cell in enumerate(cells, start=2):
        worlds.append(
            _ordinary_condition(
                spec.root,
                f"world-{cell.label}-validation.jsonl",
                f"world/{cell.label}/validation",
            )
        )
        elapsed = perf_counter() - started
        print(
            f"  [{spec.dimensions} {ordinal}/11] world/{cell.label}/validation; "
            f"grid ETA {elapsed * (11 - ordinal) / ordinal:.0f}s",
            flush=True,
        )
    controls: list[Condition] = []
    for ordinal, cell in enumerate(cells, start=7):
        controls.append(_control_condition(spec.root, cell))
        elapsed = perf_counter() - started
        print(
            f"  [{spec.dimensions} {ordinal}/11] control/{cell.label}/validation; "
            f"grid ETA {elapsed * (11 - ordinal) / ordinal:.0f}s",
            flush=True,
        )
    return GridEvidence(
        spec=spec,
        cells=cells,
        buckets=_word_buckets(spec.root),
        base=base,
        worlds=tuple(worlds),
        controls=tuple(controls),
    )


def _bucket(grid: GridEvidence, namespace: str, index: int) -> WordBucket:
    return next(
        bucket
        for bucket in grid.buckets
        if bucket.namespace == namespace and bucket.index == index
    )


def _even_examples(words: Sequence[str], count: int = 12) -> tuple[str, ...]:
    if len(words) <= count:
        return tuple(words)
    positions = tuple(round(i * (len(words) - 1) / (count - 1)) for i in range(count))
    return tuple(words[position] for position in positions)


def _blockquote(text: str) -> str:
    return "\n".join(
        ">" if not line else f"> {line}"
        for source_line in text.splitlines()
        for line in (source_line.rstrip(),)
    )


def _sample_markdown(title: str, sample: Sample, median_token_count: int) -> str:
    recipe = sample.recipe
    feature_text = ", ".join(recipe.features) if recipe.features else "none"
    control_text = (
        f"; control arm `{sample.control_arm}`" if sample.control_arm is not None else ""
    )
    return "\n".join(
        (
            f"#### {title}",
            "",
            f"- Recipe: adjective `{recipe.adjective}`, noun `{recipe.noun}`, verb "
            f"`{recipe.verb}`; features `{feature_text}`.",
            f"- Recipe cell: `N{recipe.noun_bucket} × V{recipe.verb_bucket}`"
            f"{control_text}.",
            f"- Length: {sample.index.token_count} tokens; condition lower median "
            f"{median_token_count} tokens.",
            f"- Source: `{sample.index.source}`; archive member "
            f"`{sample.index.source_member}` record {sample.index.source_index}.",
            f"- Record ID: `{sample.index.record_id}`.",
            f"- Exact-story SHA-256: `{sample.index.story_sha256}`.",
            "",
            _blockquote(sample.text),
            "",
        )
    )


def _condition_rows(grid: GridEvidence) -> str:
    conditions = (grid.base, *grid.worlds, *grid.controls)
    return "\n".join(
        f"| `{condition.label}` | {condition.occurrence_count:,} | "
        f"{condition.active_token_count:,} | {condition.median_token_count} | "
        f"{len(condition.samples)} |"
        for condition in conditions
    )


def _topology_rows(grid: GridEvidence) -> str:
    return "\n".join(
        f"| {cell.label} | `N{cell.noun_bucket} × V{cell.verb_bucket}` | "
        f"{cell.group_count:,} | {cell.active_token_count:,} |"
        for cell in grid.cells
    )


def _bucket_rows(grid: GridEvidence) -> str:
    selected = tuple(
        sorted(
            {
                (namespace, index)
                for cell in grid.cells
                for namespace, index in (
                    ("noun", cell.noun_bucket),
                    ("verb", cell.verb_bucket),
                )
            }
        )
    )
    return "\n".join(
        f"| `{'N' if namespace == 'noun' else 'V'}{index}` | {namespace} | "
        f"{len(bucket.words)} | {bucket.token_mass:,} | "
        f"{', '.join(f'`{word}`' for word in _even_examples(bucket.words))} |"
        for namespace, index in selected
        for bucket in (_bucket(grid, namespace, index),)
    )


def _grid_markdown(grid: GridEvidence) -> str:
    world_sections = []
    for cell, world, control in zip(
        grid.cells, grid.worlds, grid.controls, strict=True
    ):
        noun_bucket = _bucket(grid, "noun", cell.noun_bucket)
        verb_bucket = _bucket(grid, "verb", cell.verb_bucket)
        world_sections.append(
            "\n".join(
                (
                    f"### World {cell.label}: N{cell.noun_bucket} × V{cell.verb_bucket}",
                    "",
                    f"This world contains recipes pairing any of "
                    f"{len(noun_bucket.words)} nouns in `N{cell.noun_bucket}` with any "
                    f"of {len(verb_bucket.words)} verbs in `V{cell.verb_bucket}`. Its "
                    "matched control combines an equal-count same-noun-row arm and "
                    "same-verb-column arm from held-in validation.",
                    "",
                    _sample_markdown(
                        f"`{world.label}`", world.samples[0], world.median_token_count
                    ),
                    _sample_markdown(
                        f"`{control.label}` — same noun row",
                        control.samples[0],
                        control.median_token_count,
                    ),
                    _sample_markdown(
                        f"`{control.label}` — same verb column",
                        control.samples[1],
                        control.median_token_count,
                    ),
                )
            )
        )
    return "\n".join(
        (
            f"## {grid.spec.title}",
            "",
            f"Partition: `{grid.spec.partition_sha256}`.",
            "",
            "### Condition inventory",
            "",
            "| Validation condition | Occurrences | Active tokens | Lower-median "
            "length | Samples below |",
            "| --- | ---: | ---: | ---: | ---: |",
            _condition_rows(grid),
            "",
            "### Selected world topology",
            "",
            "| World | Physical cell | Eligible groups | Active tokens |",
            "| :---: | :---: | ---: | ---: |",
            _topology_rows(grid),
            "",
            "### What the selected buckets contain",
            "",
            "Examples are 12 evenly spaced entries from each alphabetically sorted "
            "bucket, not hand-picked themes.",
            "",
            "| Bucket | Kind | Words | Archive token mass | Deterministic examples |",
            "| :---: | :---: | ---: | ---: | --- |",
            _bucket_rows(grid),
            "",
            "### Held-in base validation",
            "",
            _sample_markdown(
                f"`{grid.base.label}`",
                grid.base.samples[0],
                grid.base.median_token_count,
            ),
            *world_sections,
        )
    )


def _render_report(grids: Sequence[GridEvidence]) -> str:
    return "\n".join(
        (
            "# TinyWorlds-P Archive v1 Validation Sample Appendix",
            "",
            "This appendix makes the archive-only calibration conditions concrete. "
            "It covers all 11 validation conditions used to compute the gap on each "
            "grid: held-in base, worlds A–E, and controls A–E. Sealed-test indexes "
            "were not read.",
            "",
            "## Deterministic selection rule",
            "",
            "For each validation index, the generator computes the lower-median "
            "document token count, orders documents by absolute distance from that "
            "median and then by normalized-story SHA-256 and record ID, and selects "
            "the first. A matched control is an equal-count mixture of two materially "
            "different arms, so the appendix selects the first candidate from each "
            "arm under that same order. This yields one base story, one story per "
            "world, and two stories per control on each grid without inspecting story "
            "semantics. Every displayed story was reconstructed from its persisted "
            "text-shard offset and checked against its exact-story SHA-256. The "
            "Markdown renderer removes only invisible trailing line whitespace after "
            "that byte-level check.",
            "",
            "The noun, verb, adjective, and feature recipe is construction evidence. "
            "The model received only the exact story text shown in the block quote; "
            "it did not receive recipes, bucket numbers, world labels, source labels, "
            "or archive coordinates.",
            "",
            *(_grid_markdown(grid) for grid in grids),
            "",
        )
    )


def main() -> None:
    """Collect both grids in parallel and write the deterministic appendix."""
    started = perf_counter()
    print(
        "[phase 1/2] Reading both grids' validation indexes in two worker "
        "processes...",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(_collect_grid, spec): spec for spec in GRID_SPECS}
        completed = {
            futures[future].partition_sha256: future.result()
            for future in as_completed(futures)
        }
    grids = tuple(completed[spec.partition_sha256] for spec in GRID_SPECS)
    print(
        f"[phase 2/2] Verified {sum(1 + 5 + 10 for _ in grids)} exact story "
        "samples; rendering appendix...",
        flush=True,
    )
    OUTPUT_PATH.write_text(_render_report(grids).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(REPOSITORY_ROOT)}.", flush=True)
    print(f"Completed in {perf_counter() - started:.1f}s.")


if __name__ == "__main__":
    main()
