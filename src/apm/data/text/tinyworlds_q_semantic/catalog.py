"""Reviewed catalog construction, sealed publication, and strict loading."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import html
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Protocol, runtime_checkable

from apm.data.text.tinyworlds_p.contracts import (
    HashedFile,
    NormalizationIdentity,
    SourceIdentity,
    TokenizerIdentity,
)
from apm.data.text.tinyworlds_q_semantic.contracts import (
    BENCHMARK_ID,
    CATALOG_FORMAT,
    SCHEMA_VERSION,
    ConceptDefinition,
    FactReviewDecision,
    RejectedFactCandidate,
    SemanticFact,
    SemanticQueryCatalog,
    SemanticQueryTemplate,
    StoryProvenance,
    canonical_json_bytes,
    record_sha256,
    registered_distractor_order,
    validate_parent_catalog_prefix,
)
from apm.data.text.tinyworlds_q_semantic.review import SemanticReviewPacket
from apm.lm.text import TextTokenizer


CATALOG_TREE_FORMAT = "tinyworlds-q-semantic-catalog-tree-v1"


@runtime_checkable
class SealedCatalogAuthorization(Protocol):
    """Structural authorization required before sealed query deserialization."""

    @property
    def catalog_sha256(self) -> str:
        """Return the frozen catalog identity."""
        ...

    @property
    def test_access_authorized(self) -> bool:
        """Return whether all model artifacts and settings were frozen first."""
        ...


@dataclass(frozen=True, slots=True)
class ValidationCatalogView:
    """Training-safe catalog view that never deserializes sealed test prompts."""

    root: Path
    catalog_sha256: str
    concepts: tuple[ConceptDefinition, ...]
    facts: tuple[SemanticFact, ...]
    templates: tuple[SemanticQueryTemplate, ...]
    reviews: tuple[FactReviewDecision, ...]
    rejected_candidates: tuple[RejectedFactCandidate, ...]
    review_packet_sha256: str
    parent_catalog_sha256: str | None
    archive_identity: SourceIdentity
    tokenizer_identity: TokenizerIdentity
    normalization: NormalizationIdentity

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        if any(template.split != "validation" for template in self.templates):
            raise ValueError("validation catalog views cannot contain sealed templates")
        expected_fact_ids = {fact.fact_id for fact in self.facts}
        counts = {
            fact_id: sum(template.fact_id == fact_id for template in self.templates)
            for fact_id in expected_fact_ids
        }
        if any(count != 3 for count in counts.values()):
            raise ValueError("validation catalog views require three templates per fact")

    @property
    def concept_ids(self) -> tuple[str, ...]:
        """Return the ordered concept manifest."""
        return tuple(concept.concept_id for concept in self.concepts)


def registered_answer_position(split: str, paraphrase_index: int) -> int:
    """Balance each fact's eight correct answers exactly across four positions."""
    limits = {"validation": 3, "test": 5}
    if split not in limits:
        raise ValueError("query split must be validation or test")
    if type(paraphrase_index) is not int or not 0 <= paraphrase_index < limits[split]:
        raise ValueError("paraphrase index lies outside its registered split")
    return (paraphrase_index + (0 if split == "validation" else 3)) % 4


def make_query_template(
    fact: SemanticFact,
    concept: ConceptDefinition,
    *,
    template_id: str,
    direction: str,
    prompt_text: str,
    distractors: tuple[str, str, str],
    split: str,
    paraphrase_index: int,
    tokenizer: TextTokenizer,
) -> SemanticQueryTemplate:
    """Compile one reviewed paraphrase and bind its exact answer token boundaries."""
    if fact.concept_id != concept.concept_id:
        raise ValueError("query template fact and concept do not match")
    if direction not in ("forward", "reverse"):
        raise ValueError("query template direction must be forward or reverse")
    if split not in ("validation", "test"):
        raise ValueError("query template split must be validation or test")
    correct_index = registered_answer_position(split, paraphrase_index)
    ordered_distractors = registered_distractor_order(
        distractors,
        split,  # type: ignore[arg-type]
        paraphrase_index,
    )
    canonical_answer = (
        fact.canonical_answer if direction == "forward" else concept.surface_forms[0]
    )
    distractor_iterator = iter(ordered_distractors)
    candidates = tuple(
        canonical_answer if index == correct_index else next(distractor_iterator)
        for index in range(4)
    )
    prompt_tokens = tokenizer.encode(prompt_text)
    combined = tuple(
        tokenizer.encode(f"{prompt_text} {candidate}") for candidate in candidates
    )
    if any(tokens[: len(prompt_tokens)] != prompt_tokens for tokens in combined):
        raise ValueError(
            "tokenizer boundary changed between the standalone prompt and answer"
        )
    grammatical_type = fact.answer_type if direction == "forward" else "concept"
    return SemanticQueryTemplate(
        template_id=template_id,
        fact_id=fact.fact_id,
        direction=direction,
        prompt_text=prompt_text,
        canonical_answer_form=canonical_answer,
        distractors=ordered_distractors,
        candidate_grammatical_types=(grammatical_type,) * 4,
        split=split,
        correct_candidate_index=correct_index,
        prompt_token_ids=prompt_tokens,
        combined_candidate_token_ids=combined,  # type: ignore[arg-type]
    )


def build_reviewed_catalog(
    *,
    review_packet: SemanticReviewPacket,
    facts: tuple[SemanticFact, ...],
    templates: tuple[SemanticQueryTemplate, ...],
    reviews: tuple[FactReviewDecision, ...],
    rejected_candidates: tuple[RejectedFactCandidate, ...],
    parent: SemanticQueryCatalog | None = None,
) -> SemanticQueryCatalog:
    """Build semantic authority only after validating it against review evidence."""
    catalog = SemanticQueryCatalog(
        concepts=review_packet.concepts,
        facts=facts,
        templates=templates,
        reviews=reviews,
        rejected_candidates=rejected_candidates,
        review_packet_sha256=review_packet.packet_sha256,
        parent_catalog_sha256=(None if parent is None else parent.catalog_sha256),
        archive_identity=review_packet.archive_identity,
    )
    validate_catalog_against_review_packet(catalog, review_packet)
    if parent is not None:
        validate_parent_catalog_prefix(catalog, parent)
    return catalog


def validate_catalog_against_review_packet(
    catalog: SemanticQueryCatalog,
    packet: SemanticReviewPacket,
) -> None:
    """Require every accepted/rejected decision to trace to construction evidence."""
    if (
        catalog.review_packet_sha256 != packet.packet_sha256
        or catalog.concepts != packet.concepts
        or catalog.archive_identity != packet.archive_identity
    ):
        raise ValueError("catalog changed its review packet or source binding")
    candidate_by_id = {candidate.candidate_id: candidate for candidate in packet.candidates}
    concept_by_id = {concept.concept_id: concept for concept in packet.concepts}
    for fact in catalog.facts:
        candidate = candidate_by_id.get(fact.source_candidate_id)
        if candidate is None:
            raise ValueError(f"fact {fact.fact_id} does not bind a review candidate")
        if (
            candidate.concept_id != fact.concept_id
            or candidate.relation_category != fact.relation_category
            or candidate.predicate not in fact.trigger_forms
            or not set(fact.supporting_story_groups).issubset(
                candidate.supporting_story_groups
            )
            or not set(fact.evidence).issubset(candidate.evidence)
        ):
            raise ValueError(f"fact {fact.fact_id} changed reviewed evidence semantics")
        concept = concept_by_id[fact.concept_id]
        if any(
            not (
                _contains_surface(item.sentence_text, concept.surface_forms)
                and _contains_surface(
                    item.sentence_text,
                    (candidate.predicate,),
                )
            )
            for item in fact.evidence
        ):
            raise ValueError(f"fact {fact.fact_id} changed authoritative evidence")
    for rejected in catalog.rejected_candidates:
        candidate = candidate_by_id.get(rejected.candidate_id)
        if candidate is None or (
            candidate.concept_id != rejected.concept_id
            or candidate.predicate != rejected.predicate
            or candidate.supporting_story_groups != rejected.evidence_group_sha256
        ):
            raise ValueError("rejected candidate changed its review-packet evidence")


def publish_catalog(
    catalog: SemanticQueryCatalog,
    output_root: str | Path,
) -> Path:
    """Atomically publish metadata and physically separate sealed test queries."""
    root = Path(output_root) / "catalog" / catalog.catalog_sha256
    if root.exists():
        _verify_catalog_tree(root, catalog.catalog_sha256)
        return root
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".catalog-", dir=root.parent))
    try:
        catalog_record = catalog.as_record()
        metadata = {key: value for key, value in catalog_record.items() if key != "templates"}
        metadata["template_order"] = [
            template.template_id for template in catalog.templates
        ]
        validation_templates = tuple(
            template for template in catalog.templates if template.split == "validation"
        )
        test_templates = tuple(
            template for template in catalog.templates if template.split == "test"
        )
        payloads = {
            "metadata.json": canonical_json_bytes(metadata),
            "validation-queries.json": canonical_json_bytes(
                {
                    "catalog_sha256": catalog.catalog_sha256,
                    "templates": [item.as_record() for item in validation_templates],
                }
            ),
            "sealed-test-queries.json": canonical_json_bytes(
                {
                    "catalog_sha256": catalog.catalog_sha256,
                    "templates": [item.as_record() for item in test_templates],
                }
            ),
        }
        audit_markdown = render_catalog_audit(catalog, include_sealed=False)
        payloads["audit.md"] = audit_markdown.encode("utf-8")
        payloads["audit.html"] = _standalone_html(
            catalog.catalog_sha256,
            audit_markdown,
        ).encode("utf-8")
        for name, payload in payloads.items():
            (staging / name).write_bytes(payload)
        files = [
            {
                "name": name,
                "sha256": sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for name, payload in sorted(payloads.items())
        ]
        manifest = {
            "catalog_sha256": catalog.catalog_sha256,
            "files": files,
            "format": CATALOG_TREE_FORMAT,
            "schema_version": SCHEMA_VERSION,
        }
        (staging / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        os.replace(staging, root)
    except BaseException:
        _remove_staging(staging)
        raise
    return root


def load_validation_catalog(directory: str | Path) -> ValidationCatalogView:
    """Strictly load only validation prompts while authenticating the sealed file."""
    root = Path(directory)
    manifest = _verify_catalog_tree(root)
    metadata = _load_json(root / "metadata.json")
    validation = _load_json(root / "validation-queries.json")
    catalog_sha256 = _string(manifest, "catalog_sha256")
    if (
        _string(metadata, "catalog_sha256") != catalog_sha256
        or _string(validation, "catalog_sha256") != catalog_sha256
    ):
        raise ValueError("validation catalog source binding changed")
    decoded = _decode_metadata(metadata)
    templates = tuple(
        _decode_template(item)
        for item in _record_list(validation, "templates")
    )
    return ValidationCatalogView(
        root=root.resolve(),
        catalog_sha256=catalog_sha256,
        templates=templates,
        **decoded,
    )


def load_sealed_catalog(
    directory: str | Path,
    authorization: SealedCatalogAuthorization,
) -> SemanticQueryCatalog:
    """Deserialize test prompts only for a matching durable frozen transaction."""
    if not isinstance(authorization, SealedCatalogAuthorization):
        raise TypeError("sealed catalog loading requires a structural authorization")
    root = Path(directory)
    manifest = _verify_catalog_tree(root)
    catalog_sha256 = _string(manifest, "catalog_sha256")
    if (
        not authorization.test_access_authorized
        or authorization.catalog_sha256 != catalog_sha256
    ):
        raise PermissionError("sealed test queries are not authorized for this catalog")
    metadata = _load_json(root / "metadata.json")
    validation = _load_json(root / "validation-queries.json")
    sealed = _load_json(root / "sealed-test-queries.json")
    if any(
        _string(record, "catalog_sha256") != catalog_sha256
        for record in (metadata, validation, sealed)
    ):
        raise ValueError("sealed catalog source binding changed")
    decoded = _decode_metadata(metadata)
    unordered_templates = tuple(
        _decode_template(item)
        for record in (validation, sealed)
        for item in _record_list(record, "templates")
    )
    template_by_id = {
        template.template_id: template for template in unordered_templates
    }
    template_order = _string_tuple(metadata, "template_order")
    if set(template_order) != set(template_by_id):
        raise ValueError("sealed catalog template order changed")
    templates = tuple(template_by_id[template_id] for template_id in template_order)
    catalog = SemanticQueryCatalog(templates=templates, **decoded)
    if catalog.catalog_sha256 != catalog_sha256:
        raise ValueError("sealed catalog content hash changed")
    return catalog


def publish_opened_sealed_audit(
    catalog: SemanticQueryCatalog,
    authorization: SealedCatalogAuthorization,
    output_directory: str | Path,
) -> tuple[Path, Path]:
    """Publish the full query audit only inside the one opened test transaction."""
    if (
        not isinstance(authorization, SealedCatalogAuthorization)
        or not authorization.test_access_authorized
        or authorization.catalog_sha256 != catalog.catalog_sha256
    ):
        raise PermissionError("full sealed audit requires the matching open transaction")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    markdown_path = output / "sealed-catalog-audit.md"
    html_path = output / "sealed-catalog-audit.html"
    markdown = render_catalog_audit(catalog, include_sealed=True)
    payloads = (
        (markdown_path, markdown.encode("utf-8")),
        (
            html_path,
            _standalone_html(catalog.catalog_sha256, markdown).encode("utf-8"),
        ),
    )
    for path, payload in payloads:
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    return markdown_path, html_path


def render_catalog_audit(
    catalog: SemanticQueryCatalog,
    *,
    include_sealed: bool,
) -> str:
    """Render facts, evidence, distractors, tokenization, and sealed commitments."""
    lines = [
        "# TinyWorlds-Q semantic catalog audit",
        "",
        f"Catalog: `{catalog.catalog_sha256}`",
        "",
        f"Review packet: `{catalog.review_packet_sha256}`",
        "",
        "The construction slice is permanently excluded from model inputs.",
        "",
    ]
    templates_by_fact = {
        fact.fact_id: tuple(
            template for template in catalog.templates if template.fact_id == fact.fact_id
        )
        for fact in catalog.facts
    }
    for concept in catalog.concepts:
        lines.extend(
            (
                f"## {concept.concept_id}",
                "",
                f"Surfaces: {', '.join(f'`{item}`' for item in concept.surface_forms)}",
                "",
            )
        )
        for fact in (item for item in catalog.facts if item.concept_id == concept.concept_id):
            review = next(item for item in catalog.reviews if item.fact_id == fact.fact_id)
            lines.extend(
                (
                    f"### {fact.fact_id}",
                    "",
                    f"Relation: `{fact.relation_category}`; answer type: `{fact.answer_type}`  ",
                    f"Answer: `{fact.canonical_answer}`  ",
                    f"Accepted: {', '.join(f'`{item}`' for item in fact.accepted_forms)}  ",
                    f"Triggers: {', '.join(f'`{item}`' for item in fact.trigger_forms)}  ",
                    f"Reviewer: {review.reviewer} at {review.reviewed_at}  ",
                    "Approvals: truth=true; answer forms=true; trigger closure=true; "
                    "distractors=true; evidence=true",
                    "",
                    "Evidence:",
                    "",
                )
            )
            lines.extend(
                f"- `{item.group_sha256}` / `{item.record_id}` sentence {item.sentence_index}: {item.sentence_text}"
                for item in fact.evidence
            )
            lines.append("")
            for template in templates_by_fact[fact.fact_id]:
                if template.split == "test" and not include_sealed:
                    lines.extend(
                        (
                            f"- sealed template `{template.template_id}`: "
                            f"commitment `{record_sha256(template.as_record())}`",
                            "",
                        )
                    )
                    continue
                lines.extend(
                    (
                        f"- `{template.template_id}` ({template.split}, {template.direction})",
                        f"  - prompt: {template.prompt_text}",
                        f"  - candidates: {template.candidate_answer_forms}",
                        f"  - correct position: {template.correct_candidate_index}",
                        f"  - prompt tokens: {template.prompt_token_ids}",
                        f"  - answer tokens: {template.answer_token_ids}",
                    )
                )
            lines.append("")
    lines.extend(("## Rejected candidates", ""))
    for item in catalog.rejected_candidates:
        lines.extend(
            (
                f"- `{item.candidate_id}` ({item.concept_id}, `{item.predicate}`): "
                f"{item.reason}",
                f"  - reviewer: {item.reviewer}",
                "  - reviewed evidence groups: "
                + ", ".join(f"`{group}`" for group in item.evidence_group_sha256),
            )
        )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _decode_metadata(record: dict[str, object]) -> dict[str, object]:
    expected = {
        "archive_identity",
        "benchmark_id",
        "catalog_sha256",
        "concepts",
        "construction",
        "facts",
        "format",
        "normalization",
        "parent_catalog_sha256",
        "rejected_candidates",
        "review_packet_sha256",
        "reviews",
        "schema_version",
        "template_order",
        "tokenizer_identity",
    }
    if set(record) != expected or record.get("benchmark_id") != BENCHMARK_ID or record.get("format") != CATALOG_FORMAT or record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("catalog metadata format changed")
    parent_value = record["parent_catalog_sha256"]
    if parent_value is not None and type(parent_value) is not str:
        raise ValueError("parent catalog identity must be text or null")
    return {
        "concepts": tuple(_decode_concept(item) for item in _record_list(record, "concepts")),
        "facts": tuple(_decode_fact(item) for item in _record_list(record, "facts")),
        "reviews": tuple(_decode_review(item) for item in _record_list(record, "reviews")),
        "rejected_candidates": tuple(
            _decode_rejected(item) for item in _record_list(record, "rejected_candidates")
        ),
        "review_packet_sha256": _string(record, "review_packet_sha256"),
        "parent_catalog_sha256": parent_value,
        "archive_identity": _decode_source(_mapping(record, "archive_identity")),
        "tokenizer_identity": _decode_tokenizer(_mapping(record, "tokenizer_identity")),
        "normalization": _decode_normalization(_mapping(record, "normalization")),
    }


def _decode_concept(record: dict[str, object]) -> ConceptDefinition:
    return ConceptDefinition(
        concept_id=_string(record, "concept_id"),
        surface_forms=_string_tuple(record, "surface_forms"),
    )


def _decode_provenance(record: dict[str, object]) -> StoryProvenance:
    return StoryProvenance(
        group_sha256=_string(record, "group_sha256"),
        story_sha256=_string(record, "story_sha256"),
        record_id=_string(record, "record_id"),
        source_member=_string(record, "source_member"),
        source_index=_integer(record, "source_index"),
        sentence_index=_integer(record, "sentence_index"),
        sentence_text=_string(record, "sentence_text"),
    )


def _decode_fact(record: dict[str, object]) -> SemanticFact:
    return SemanticFact(
        fact_id=_string(record, "fact_id"),
        source_candidate_id=_string(record, "source_candidate_id"),
        concept_id=_string(record, "concept_id"),
        relation_category=_string(record, "relation_category"),
        answer_type=_string(record, "answer_type"),
        canonical_answer=_string(record, "canonical_answer"),
        accepted_forms=_string_tuple(record, "accepted_forms"),
        trigger_forms=_string_tuple(record, "trigger_forms"),
        supporting_story_groups=_string_tuple(record, "supporting_story_groups"),
        evidence=tuple(
            _decode_provenance(item) for item in _record_list(record, "evidence")
        ),
    )


def _decode_template(record: dict[str, object]) -> SemanticQueryTemplate:
    distractors = _string_tuple(record, "distractors")
    grammatical = _string_tuple(record, "candidate_grammatical_types")
    combined_raw = _list(record, "combined_candidate_token_ids")
    combined = tuple(_integer_tuple(value, "combined candidate tokens") for value in combined_raw)
    return SemanticQueryTemplate(
        template_id=_string(record, "template_id"),
        fact_id=_string(record, "fact_id"),
        direction=_string(record, "direction"),  # type: ignore[arg-type]
        prompt_text=_string(record, "prompt_text"),
        canonical_answer_form=_string(record, "canonical_answer_form"),
        distractors=distractors,  # type: ignore[arg-type]
        candidate_grammatical_types=grammatical,  # type: ignore[arg-type]
        split=_string(record, "split"),  # type: ignore[arg-type]
        correct_candidate_index=_integer(record, "correct_candidate_index"),
        prompt_token_ids=_integer_tuple(record.get("prompt_token_ids"), "prompt tokens"),
        combined_candidate_token_ids=combined,  # type: ignore[arg-type]
    )


def _decode_review(record: dict[str, object]) -> FactReviewDecision:
    return FactReviewDecision(
        fact_id=_string(record, "fact_id"),
        reviewer=_string(record, "reviewer"),
        reviewed_at=_string(record, "reviewed_at"),
        truth_approved=_boolean(record, "truth_approved"),
        answer_forms_approved=_boolean(record, "answer_forms_approved"),
        trigger_closure_approved=_boolean(record, "trigger_closure_approved"),
        distractors_approved=_boolean(record, "distractors_approved"),
        evidence_approved=_boolean(record, "evidence_approved"),
    )


def _decode_rejected(record: dict[str, object]) -> RejectedFactCandidate:
    return RejectedFactCandidate(
        candidate_id=_string(record, "candidate_id"),
        concept_id=_string(record, "concept_id"),
        predicate=_string(record, "predicate"),
        reason=_string(record, "reason"),
        reviewer=_string(record, "reviewer"),
        evidence_group_sha256=_string_tuple(record, "evidence_group_sha256"),
    )


def _decode_source(record: dict[str, object]) -> SourceIdentity:
    return SourceIdentity(
        dataset_id=_string(record, "dataset_id"),
        revision=_string(record, "revision"),
        filename=_string(record, "filename"),
        size_bytes=_integer(record, "size_bytes"),
        sha256=_string(record, "sha256"),
    )


def _decode_tokenizer(record: dict[str, object]) -> TokenizerIdentity:
    files = tuple(
        HashedFile(
            name=_string(item, "name"),
            size_bytes=_integer(item, "size_bytes"),
            sha256=_string(item, "sha256"),
        )
        for item in _record_list(record, "files")
    )
    return TokenizerIdentity(
        kind=_string(record, "kind"),
        identifier=_string(record, "identifier"),
        revision=_string(record, "revision"),
        vocab_size=_integer(record, "vocab_size"),
        files=files,
    )


def _decode_normalization(record: dict[str, object]) -> NormalizationIdentity:
    return NormalizationIdentity(
        version=_string(record, "version"),
        unicode_form=_string(record, "unicode_form"),  # type: ignore[arg-type]
        case_folding=_boolean(record, "case_folding"),
        whitespace_collapse=_boolean(record, "whitespace_collapse"),
        canonical_straight_quotes=_boolean(record, "canonical_straight_quotes"),
    )


def _verify_catalog_tree(
    root: Path,
    expected_catalog_sha256: str | None = None,
) -> dict[str, object]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifest = _load_json(root / "manifest.json")
    if (
        manifest.get("format") != CATALOG_TREE_FORMAT
        or manifest.get("schema_version") != SCHEMA_VERSION
        or (expected_catalog_sha256 is not None and manifest.get("catalog_sha256") != expected_catalog_sha256)
    ):
        raise ValueError("catalog tree manifest changed")
    if root.name != _string(manifest, "catalog_sha256"):
        raise ValueError("catalog directory name does not match its content identity")
    file_records = _record_list(manifest, "files")
    expected_names = {"manifest.json", *(_string(item, "name") for item in file_records)}
    actual_names = {path.name for path in root.iterdir() if path.is_file()}
    if actual_names != expected_names or any(path.is_dir() for path in root.iterdir()):
        raise ValueError("catalog tree entries changed")
    for item in file_records:
        path = root / _string(item, "name")
        payload = path.read_bytes()
        if len(payload) != _integer(item, "size_bytes") or sha256(payload).hexdigest() != _string(item, "sha256"):
            raise ValueError(f"catalog file changed: {path.name}")
    return manifest


def _contains_surface(text: str, surfaces: tuple[str, ...]) -> bool:
    normalized = text.casefold()
    return any(
        re.search(rf"(?<!\w){re.escape(surface)}(?!\w)", normalized) is not None
        for surface in surfaces
    )


def _standalone_html(catalog_sha256: str, markdown: str) -> str:
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>TinyWorlds-Q catalog audit</title>"
        "<style>body{font:15px/1.5 system-ui;max-width:1100px;margin:2rem auto;"
        "padding:0 1rem}pre{white-space:pre-wrap;overflow-wrap:anywhere;"
        "background:#f5f7f9;padding:1.25rem;border-radius:8px}</style></head>"
        f"<body data-catalog-sha256=\"{catalog_sha256}\"><pre>"
        f"{html.escape(markdown)}</pre></body></html>\n"
    )


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid catalog JSON {path.name}: {error}") from error
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise ValueError(f"catalog JSON is not one canonical record: {path.name}")
    return value


def _mapping(record: dict[str, object], field: str) -> dict[str, object]:
    value = record.get(field)
    if type(value) is not dict:
        raise ValueError(f"catalog {field} must be an object")
    return value


def _list(record: dict[str, object], field: str) -> list[object]:
    value = record.get(field)
    if type(value) is not list:
        raise ValueError(f"catalog {field} must be an array")
    return value


def _record_list(record: dict[str, object], field: str) -> tuple[dict[str, object], ...]:
    values = _list(record, field)
    if any(type(value) is not dict for value in values):
        raise ValueError(f"catalog {field} must contain objects")
    return tuple(values)  # type: ignore[arg-type]


def _string(record: dict[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise ValueError(f"catalog {field} must be nonempty text")
    return value


def _integer(record: dict[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise ValueError(f"catalog {field} must be a nonnegative integer")
    return value


def _boolean(record: dict[str, object], field: str) -> bool:
    value = record.get(field)
    if type(value) is not bool:
        raise ValueError(f"catalog {field} must be boolean")
    return value


def _string_tuple(record: dict[str, object], field: str) -> tuple[str, ...]:
    values = _list(record, field)
    if any(type(value) is not str for value in values):
        raise ValueError(f"catalog {field} must contain text")
    return tuple(values)  # type: ignore[arg-type]


def _integer_tuple(value: object, label: str) -> tuple[int, ...]:
    if type(value) is not list or any(type(item) is not int or item < 0 for item in value):
        raise ValueError(f"{label} must contain nonnegative integers")
    return tuple(value)


def _remove_staging(staging: Path) -> None:
    if not staging.exists():
        return
    for path in sorted(staging.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    staging.rmdir()


__all__ = [
    "CATALOG_TREE_FORMAT",
    "SealedCatalogAuthorization",
    "ValidationCatalogView",
    "build_reviewed_catalog",
    "load_sealed_catalog",
    "load_validation_catalog",
    "make_query_template",
    "publish_catalog",
    "publish_opened_sealed_audit",
    "registered_answer_position",
    "render_catalog_audit",
    "validate_catalog_against_review_packet",
]
