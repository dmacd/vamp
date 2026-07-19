"""Blinded human-audit packets and approval gates for TinyWorlds-v2."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import hashlib
import html
import json
import math
import re


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class AuditAllocationError(ValueError):
    """The requested balanced audit has no globally distinct pair allocation."""


class AuditSourceKind(str, Enum):
    """Whether an audit item is genuine or model generated."""

    REFERENCE = "reference"
    GENERATED = "generated"


class AuditSourceGuess(str, Enum):
    """A blinded rater's source guess."""

    REFERENCE = "reference"
    GENERATED = "generated"


@dataclass(frozen=True, slots=True)
class AuditSourceRecord:
    """One candidate story before its source identity is hidden."""

    source_id: str
    pair_id: str
    story_text: str
    source_prompt: str
    token_count: int
    base_normalized_nll: float
    automated_style_scores: tuple[tuple[str, float], ...]
    source_kind: AuditSourceKind
    route_id: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.source_id, "source_id")
        _require_text(self.pair_id, "pair_id")
        _require_text(self.story_text, "story_text")
        _require_text(self.source_prompt, "source_prompt")
        _validate_blinded_metrics(
            self.token_count,
            self.base_normalized_nll,
            self.automated_style_scores,
        )
        if type(self.source_kind) is not AuditSourceKind:
            raise TypeError("source_kind must be an AuditSourceKind")
        if self.source_kind is AuditSourceKind.REFERENCE and self.route_id is not None:
            raise ValueError("reference audit records cannot have route IDs")
        if self.source_kind is AuditSourceKind.GENERATED:
            _require_text(self.route_id, "generated route_id")


@dataclass(frozen=True, slots=True)
class BlindedAuditItem:
    """One opaque story exposed to the human rater."""

    item_id: str
    story_text: str
    source_prompt: str
    token_count: int
    base_normalized_nll: float
    automated_style_scores: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        _require_text(self.item_id, "item_id")
        _require_text(self.story_text, "story_text")
        _require_text(self.source_prompt, "source_prompt")
        _validate_blinded_metrics(
            self.token_count,
            self.base_normalized_nll,
            self.automated_style_scores,
        )


@dataclass(frozen=True, slots=True)
class AuditKeyEntry:
    """Hidden source identity corresponding to one blinded item."""

    item_id: str
    source_id: str
    pair_id: str
    source_kind: AuditSourceKind
    route_id: str | None

    def __post_init__(self) -> None:
        _require_text(self.item_id, "item_id")
        _require_text(self.source_id, "source_id")
        _require_text(self.pair_id, "pair_id")
        if type(self.source_kind) is not AuditSourceKind:
            raise TypeError("source_kind must be an AuditSourceKind")
        if self.source_kind is AuditSourceKind.REFERENCE and self.route_id is not None:
            raise ValueError("reference key entries cannot have route IDs")
        if self.source_kind is AuditSourceKind.GENERATED:
            _require_text(self.route_id, "generated route_id")


@dataclass(frozen=True, slots=True)
class BlindedAuditPacket:
    """The source-free packet displayed to a human rater."""

    items: tuple[BlindedAuditItem, ...]
    audit_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.items) is not tuple
            or any(type(item) is not BlindedAuditItem for item in self.items)
            or not self.items
            or len({item.item_id for item in self.items}) != len(self.items)
        ):
            raise ValueError("audit packet items must be nonempty and uniquely identified")
        _require_sha256(self.audit_sha256, "audit_sha256")


@dataclass(frozen=True, slots=True)
class BlindedAuditKey:
    """Separate source key that must remain hidden until rating export."""

    entries: tuple[AuditKeyEntry, ...]
    audit_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.entries) is not tuple
            or any(type(entry) is not AuditKeyEntry for entry in self.entries)
            or not self.entries
            or len({entry.item_id for entry in self.entries}) != len(self.entries)
        ):
            raise ValueError("audit key entries must be nonempty and uniquely identified")
        _require_sha256(self.audit_sha256, "audit_sha256")


@dataclass(frozen=True, slots=True)
class AuditDecision:
    """All required ratings for one blinded story."""

    item_id: str
    ts_like_accepted: bool
    simplicity_rating: int
    coherence_rating: int
    source_guess: AuditSourceGuess

    def __post_init__(self) -> None:
        _require_text(self.item_id, "item_id")
        if type(self.ts_like_accepted) is not bool:
            raise TypeError("ts_like_accepted must be a bool")
        if (
            type(self.simplicity_rating) is not int
            or self.simplicity_rating not in range(1, 6)
        ):
            raise ValueError("simplicity_rating must be between 1 and 5")
        if (
            type(self.coherence_rating) is not int
            or self.coherence_rating not in range(1, 6)
        ):
            raise ValueError("coherence_rating must be between 1 and 5")
        if type(self.source_guess) is not AuditSourceGuess:
            raise TypeError("source_guess must be an AuditSourceGuess")


@dataclass(frozen=True, slots=True)
class AuditDecisionSet:
    """A complete decision export bound to one exact blinded packet."""

    audit_sha256: str
    decisions: tuple[AuditDecision, ...]
    decision_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.audit_sha256, "audit_sha256")
        _require_sha256(self.decision_sha256, "decision_sha256")


@dataclass(frozen=True, slots=True)
class AuditEvaluation:
    """Human-gate metrics computed only after joining the hidden key."""

    audit_sha256: str
    decision_sha256: str
    reference_acceptance_rate: float
    generated_acceptance_rate: float
    route_acceptance_rates: tuple[tuple[str, float], ...]
    source_discrimination_accuracy: float
    mean_simplicity_rating: float
    mean_coherence_rating: float
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether all human audit gates passed."""
        return not self.failures


@dataclass(frozen=True, slots=True)
class AuditApproval:
    """Explicit human approval bound to exact audit and decision digests."""

    audit_sha256: str
    decision_sha256: str
    approved_by: str
    approved_at_utc: str
    approved: bool

    def __post_init__(self) -> None:
        _require_sha256(self.audit_sha256, "audit_sha256")
        _require_sha256(self.decision_sha256, "decision_sha256")
        _require_text(self.approved_by, "approved_by")
        _require_text(self.approved_at_utc, "approved_at_utc")
        if type(self.approved) is not bool:
            raise TypeError("approved must be a bool")


def build_blinded_audit(
    reference_records: tuple[AuditSourceRecord, ...],
    generated_records: tuple[AuditSourceRecord, ...],
    *,
    finalist_order: tuple[str, ...],
    seed: str,
    reference_count: int = 100,
    generated_count: int = 100,
) -> tuple[BlindedAuditPacket, BlindedAuditKey]:
    """Build a hash-ranked, balanced, and fully blinded human audit."""
    _require_text(seed, "seed")
    if not finalist_order or len(set(finalist_order)) != len(finalist_order):
        raise ValueError("finalist_order must contain unique route IDs")
    if any(record.source_kind is not AuditSourceKind.REFERENCE for record in reference_records):
        raise ValueError("reference_records must contain only reference sources")
    if any(record.source_kind is not AuditSourceKind.GENERATED for record in generated_records):
        raise ValueError("generated_records must contain only generated sources")
    if len({record.source_id for record in reference_records + generated_records}) != len(
        reference_records + generated_records
    ):
        raise ValueError("audit source IDs must be globally unique")
    if type(reference_count) is not int or reference_count <= 0:
        raise ValueError("reference_count must be positive")
    if type(generated_count) is not int or generated_count <= 0:
        raise ValueError("generated_count must be positive")
    if reference_count != generated_count:
        raise ValueError("the blinded audit requires one matched reference per generation")
    references_by_pair = {record.pair_id: record for record in reference_records}
    if len(references_by_pair) != len(reference_records):
        raise ValueError("reference audit records must have unique pair IDs")
    by_route = {
        route_id: tuple(
            record for record in generated_records if record.route_id == route_id
        )
        for route_id in finalist_order
    }
    if any(
        len({record.pair_id for record in records}) != len(records)
        for records in by_route.values()
    ):
        raise ValueError("each route must contain at most one generation per pair")
    unknown_routes = {
        record.route_id for record in generated_records
    } - set(finalist_order)
    if unknown_routes:
        raise ValueError("generated audit records include an undeclared route")
    quotient, remainder = divmod(generated_count, len(finalist_order))
    route_counts = {
        route_id: quotient + (index < remainder)
        for index, route_id in enumerate(finalist_order)
    }
    selected_generated = _select_balanced_generations(
        finalist_order,
        by_route,
        route_counts,
        references_by_pair=frozenset(references_by_pair),
        used_pair_ids=frozenset(),
        seed=seed,
    )
    selected_references = tuple(
        references_by_pair[record.pair_id] for record in selected_generated
    )
    if any(
        reference.source_prompt != generated.source_prompt
        for reference, generated in zip(
            selected_references, selected_generated, strict=True
        )
    ):
        raise ValueError("matched reference and generated prompts must be identical")
    selected = selected_references + selected_generated
    blinded = tuple(
        (
            "audit-"
            + hashlib.sha256(
                f"{seed}\0opaque\0{record.source_id}".encode("utf-8")
            ).hexdigest()[:20],
            record,
        )
        for record in selected
    )
    if len({item_id for item_id, _ in blinded}) != len(blinded):
        raise ValueError("opaque audit ID collision")
    ordered = tuple(
        sorted(
            blinded,
            key=lambda pair: (
                hashlib.sha256(
                    f"{seed}\0audit-order\0{pair[0]}".encode("utf-8")
                ).digest(),
                pair[0],
            ),
        )
    )
    items = tuple(
        BlindedAuditItem(
            item_id=item_id,
            story_text=record.story_text,
            source_prompt=record.source_prompt,
            token_count=record.token_count,
            base_normalized_nll=record.base_normalized_nll,
            automated_style_scores=record.automated_style_scores,
        )
        for item_id, record in ordered
    )
    entries = tuple(
        AuditKeyEntry(
            item_id=item_id,
            source_id=record.source_id,
            pair_id=record.pair_id,
            source_kind=record.source_kind,
            route_id=record.route_id,
        )
        for item_id, record in ordered
    )
    audit_sha256 = _audit_digest(items, entries)
    return (
        BlindedAuditPacket(items, audit_sha256),
        BlindedAuditKey(entries, audit_sha256),
    )


def validate_audit_pair(packet: BlindedAuditPacket, key: BlindedAuditKey) -> None:
    """Reject reordered, missing, or tampered packet/key pairs."""
    if packet.audit_sha256 != key.audit_sha256:
        raise ValueError("packet and key audit digests differ")
    if tuple(item.item_id for item in packet.items) != tuple(
        entry.item_id for entry in key.entries
    ):
        raise ValueError("packet and key item order differs")
    pair_counts = Counter(entry.pair_id for entry in key.entries)
    pair_kind_counts = Counter(
        (entry.pair_id, entry.source_kind) for entry in key.entries
    )
    if any(
        count != 2
        or pair_kind_counts[(pair_id, AuditSourceKind.REFERENCE)] != 1
        or pair_kind_counts[(pair_id, AuditSourceKind.GENERATED)] != 1
        for pair_id, count in pair_counts.items()
    ):
        raise ValueError("every audit pair must contain one reference and one generation")
    if _audit_digest(packet.items, key.entries) != packet.audit_sha256:
        raise ValueError("packet or key content does not match the audit digest")


def build_decision_set(
    packet: BlindedAuditPacket,
    decisions: tuple[AuditDecision, ...],
) -> AuditDecisionSet:
    """Validate a complete rating export and bind it to the packet digest."""
    packet_ids = tuple(item.item_id for item in packet.items)
    decision_ids = tuple(decision.item_id for decision in decisions)
    if len(set(decision_ids)) != len(decision_ids):
        raise ValueError("audit decisions must have unique item IDs")
    if set(decision_ids) != set(packet_ids):
        raise ValueError("every audit item must have exactly one decision")
    by_id = {decision.item_id: decision for decision in decisions}
    ordered = tuple(by_id[item_id] for item_id in packet_ids)
    digest = _decision_digest(packet.audit_sha256, ordered)
    return AuditDecisionSet(packet.audit_sha256, ordered, digest)


def evaluate_blinded_audit(
    packet: BlindedAuditPacket,
    key: BlindedAuditKey,
    decision_set: AuditDecisionSet,
    *,
    selectable_route_ids: tuple[str, ...],
) -> AuditEvaluation:
    """Reveal sources after rating and evaluate every fixed human gate."""
    validate_audit_pair(packet, key)
    _validate_decision_set(packet, decision_set)
    if not selectable_route_ids or len(set(selectable_route_ids)) != len(
        selectable_route_ids
    ):
        raise ValueError("selectable route IDs must be nonempty and unique")
    entries = {entry.item_id: entry for entry in key.entries}
    generated_routes = {
        entry.route_id
        for entry in key.entries
        if entry.source_kind is AuditSourceKind.GENERATED
    }
    if not set(selectable_route_ids).issubset(generated_routes):
        raise ValueError("every selectable route must occur in the audit key")
    joined = tuple((decision, entries[decision.item_id]) for decision in decision_set.decisions)
    references = tuple(
        pair for pair in joined if pair[1].source_kind is AuditSourceKind.REFERENCE
    )
    generated = tuple(
        pair for pair in joined if pair[1].source_kind is AuditSourceKind.GENERATED
    )
    acceptance_rate = lambda pairs: sum(pair[0].ts_like_accepted for pair in pairs) / len(pairs)
    reference_acceptance = acceptance_rate(references)
    generated_acceptance = acceptance_rate(generated)
    route_rates = tuple(
        (
            route_id,
            acceptance_rate(tuple(pair for pair in generated if pair[1].route_id == route_id)),
        )
        for route_id in selectable_route_ids
    )
    source_accuracy = sum(
        decision.source_guess.value == entry.source_kind.value
        for decision, entry in joined
    ) / len(joined)
    failures = tuple(
        name
        for passed, name in (
            (generated_acceptance >= 0.85, "generated_acceptance_rate"),
            (
                generated_acceptance >= reference_acceptance - 0.10,
                "generated_reference_acceptance_gap",
            ),
            (
                all(rate >= 0.80 for _, rate in route_rates),
                "per_route_acceptance_rate",
            ),
            (source_accuracy <= 0.65, "source_discrimination_accuracy"),
        )
        if not passed
    )
    return AuditEvaluation(
        audit_sha256=packet.audit_sha256,
        decision_sha256=decision_set.decision_sha256,
        reference_acceptance_rate=reference_acceptance,
        generated_acceptance_rate=generated_acceptance,
        route_acceptance_rates=route_rates,
        source_discrimination_accuracy=source_accuracy,
        mean_simplicity_rating=sum(item.simplicity_rating for item, _ in joined)
        / len(joined),
        mean_coherence_rating=sum(item.coherence_rating for item, _ in joined)
        / len(joined),
        failures=failures,
    )


def select_human_approved_route(
    evaluation: AuditEvaluation,
    *,
    qualified_route_ids: tuple[str, ...],
    projected_costs: tuple[tuple[str, float], ...],
) -> str:
    """Choose the cheapest qualified route within ten points of best acceptance."""
    if not evaluation.passed:
        raise ValueError("a route cannot be selected from a failed human audit")
    rates = dict(evaluation.route_acceptance_rates)
    costs = dict(projected_costs)
    if len(costs) != len(projected_costs):
        raise ValueError("projected route costs must have unique route IDs")
    if not qualified_route_ids or not set(qualified_route_ids).issubset(rates):
        raise ValueError("qualified routes must all have human acceptance rates")
    if len(set(qualified_route_ids)) != len(qualified_route_ids):
        raise ValueError("qualified route IDs must be unique")
    if not set(qualified_route_ids).issubset(costs):
        raise ValueError("qualified routes must all have projected costs")
    if any(not math.isfinite(costs[route_id]) or costs[route_id] < 0.0 for route_id in qualified_route_ids):
        raise ValueError("projected costs must be finite and nonnegative")
    best_rate = max(rates[route_id] for route_id in qualified_route_ids)
    eligible = tuple(
        route_id
        for route_id in qualified_route_ids
        if rates[route_id] >= best_rate - 0.10
    )
    order = {route_id: index for index, route_id in enumerate(qualified_route_ids)}
    return min(eligible, key=lambda route_id: (costs[route_id], order[route_id]))


def validate_audit_approval(
    packet: BlindedAuditPacket,
    key: BlindedAuditKey,
    decision_set: AuditDecisionSet,
    evaluation: AuditEvaluation,
    approval: AuditApproval,
) -> None:
    """Unlock the next phase only for an explicit approval of exact passing evidence."""
    validate_audit_pair(packet, key)
    _validate_decision_set(packet, decision_set)
    recomputed = evaluate_blinded_audit(
        packet,
        key,
        decision_set,
        selectable_route_ids=tuple(
            route_id for route_id, _ in evaluation.route_acceptance_rates
        ),
    )
    if recomputed != evaluation:
        raise ValueError("audit evaluation does not match the packet decisions")
    if not evaluation.passed:
        raise ValueError("failed human audit evidence cannot be approved")
    expected = (packet.audit_sha256, decision_set.decision_sha256)
    if (evaluation.audit_sha256, evaluation.decision_sha256) != expected:
        raise ValueError("audit evaluation digests do not match the evidence")
    if (approval.audit_sha256, approval.decision_sha256) != expected:
        raise ValueError("approval digests do not match the evidence")
    if not approval.approved:
        raise ValueError("audit approval must explicitly set approved=true")


def render_audit_html(packet: BlindedAuditPacket) -> str:
    """Render a self-contained blinded form that exports JSON ratings."""
    item_payload = json.dumps(
        [
            {
                "item_id": item.item_id,
                "story_text": item.story_text,
                "source_prompt": item.source_prompt,
                "token_count": item.token_count,
                "base_normalized_nll": item.base_normalized_nll,
                "automated_style_scores": list(item.automated_style_scores),
            }
            for item in packet.items
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    escaped_digest = html.escape(packet.audit_sha256)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>TinyWorlds-v2 blinded audit</title>
<style>
body{{font:16px system-ui;max-width:900px;margin:2rem auto;padding:0 1rem}}
.item{{border-top:1px solid #bbb;padding:1rem 0}} .story,.prompt{{white-space:pre-wrap}}
.metrics{{color:#444}}
label{{margin-right:1rem}} select{{margin-right:1.5rem}}
</style></head><body>
<h1>TinyWorlds-v2 blinded audit</h1>
<p>Audit digest: <code>{escaped_digest}</code></p>
<p>Rate every story before exporting. Source identities are not embedded here.</p>
<div id="items"></div><button id="export" type="button">Export decisions</button>
<script>
const auditSha256={json.dumps(packet.audit_sha256)};
const items={item_payload};
const root=document.getElementById('items');
const options=(values)=>values.map(v=>`<option value="${{v}}">${{v}}</option>`).join('');
items.forEach((item,index)=>{{
  const node=document.createElement('section'); node.className='item';
  node.innerHTML=`<h2>Story ${{index+1}}</h2><h3>Source prompt</h3><div class="prompt"></div>
  <h3>Story</h3><div class="story"></div><p class="metrics"></p>
  <label>TS-like <select data-field="accepted"><option value="">Choose</option>
  <option value="true">accept</option><option value="false">reject</option></select></label>
  <label>Simplicity <select data-field="simplicity"><option value="">Choose</option>${{options([1,2,3,4,5])}}</select></label>
  <label>Coherence <select data-field="coherence"><option value="">Choose</option>${{options([1,2,3,4,5])}}</select></label>
  <label>Source guess <select data-field="guess"><option value="">Choose</option>
  <option value="reference">genuine</option><option value="generated">generated</option></select></label>`;
  node.querySelector('.prompt').textContent=item.source_prompt;
  node.querySelector('.story').textContent=item.story_text;
  node.querySelector('.metrics').textContent=`Tokens: ${{item.token_count}} · Base NLL: ${{item.base_normalized_nll.toFixed(4)}} · `+
    item.automated_style_scores.map(pair=>`${{pair[0]}}: ${{pair[1].toFixed(3)}}`).join(' · ');
  node.dataset.itemId=item.item_id;
  root.appendChild(node);
}});
document.getElementById('export').onclick=()=>{{
  const decisions=[...document.querySelectorAll('.item')].map(node=>({{
    coherence_rating:Number(node.querySelector('[data-field=coherence]').value),
    item_id:node.dataset.itemId,
    simplicity_rating:Number(node.querySelector('[data-field=simplicity]').value),
    source_guess:node.querySelector('[data-field=guess]').value,
    ts_like_accepted:node.querySelector('[data-field=accepted]').value==='true'
  }}));
  if(decisions.some(d=>!d.simplicity_rating||!d.coherence_rating||!d.source_guess||
      ![...document.querySelectorAll('.item')][decisions.indexOf(d)].querySelector('[data-field=accepted]').value)){{
    alert('Please complete every rating.'); return;
  }}
  const blob=new Blob([JSON.stringify({{audit_sha256:auditSha256,decisions:decisions}})],{{type:'application/json'}});
  const link=document.createElement('a'); link.href=URL.createObjectURL(blob);
  link.download='audit_decisions.json'; link.click(); URL.revokeObjectURL(link.href);
}};
</script></body></html>
"""


def _validate_decision_set(
    packet: BlindedAuditPacket, decision_set: AuditDecisionSet
) -> None:
    if decision_set.audit_sha256 != packet.audit_sha256:
        raise ValueError("decision set belongs to another audit")
    if tuple(item.item_id for item in packet.items) != tuple(
        decision.item_id for decision in decision_set.decisions
    ):
        raise ValueError("decision order or membership does not match the audit")
    if _decision_digest(packet.audit_sha256, decision_set.decisions) != decision_set.decision_sha256:
        raise ValueError("decision content does not match its digest")


def _audit_digest(
    items: tuple[BlindedAuditItem, ...], entries: tuple[AuditKeyEntry, ...]
) -> str:
    payload = [
        {
            "item_id": item.item_id,
            "story_text": item.story_text,
            "source_prompt": item.source_prompt,
            "token_count": item.token_count,
            "base_normalized_nll": item.base_normalized_nll,
            "automated_style_scores": [list(score) for score in item.automated_style_scores],
            "source_id": entry.source_id,
            "pair_id": entry.pair_id,
            "source_kind": entry.source_kind.value,
            "route_id": entry.route_id,
        }
        for item, entry in zip(items, entries, strict=True)
    ]
    return _canonical_digest(payload)


def _decision_digest(
    audit_sha256: str, decisions: tuple[AuditDecision, ...]
) -> str:
    return _canonical_digest(
        {
            "audit_sha256": audit_sha256,
            "decisions": [
                {
                    "item_id": item.item_id,
                    "ts_like_accepted": item.ts_like_accepted,
                    "simplicity_rating": item.simplicity_rating,
                    "coherence_rating": item.coherence_rating,
                    "source_guess": item.source_guess.value,
                }
                for item in decisions
            ],
        }
    )


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _select_balanced_generations(
    remaining_routes: tuple[str, ...],
    records_by_route: dict[str, tuple[AuditSourceRecord, ...]],
    route_counts: dict[str, int],
    *,
    references_by_pair: frozenset[str],
    used_pair_ids: frozenset[str],
    seed: str,
) -> tuple[AuditSourceRecord, ...]:
    if not remaining_routes:
        return ()
    eligible_by_route: dict[str, tuple[AuditSourceRecord, ...]] = {}
    for route_id in remaining_routes:
        eligible = tuple(
            sorted(
                (
                    record
                    for record in records_by_route[route_id]
                    if record.pair_id not in used_pair_ids
                    and record.pair_id in references_by_pair
                ),
                key=lambda item: (
                    hashlib.sha256(
                        f"{seed}\0audit-generated:{route_id}\0{item.source_id}".encode(
                            "utf-8"
                        )
                    ).digest(),
                    item.source_id,
                ),
            )
        )
        required = route_counts[route_id]
        if len(eligible) < required:
            raise AuditAllocationError(
                f"audit route {route_id!r} requires {required} matched pairs, "
                f"found {len(eligible)}"
            )
        eligible_by_route[route_id] = eligible

    # Expand route quotas into a small bipartite matching problem.  The
    # deterministic augmenting-path pass is complete: it can rearrange earlier
    # selections whenever a later, more constrained route needs their pair.
    # The previous route-greedy selection could reject feasible audits.
    slots = tuple(
        (route_id, slot_index)
        for route_id in remaining_routes
        for slot_index in range(route_counts[route_id])
    )
    pair_to_slot: dict[str, tuple[str, int]] = {}
    slot_to_pair: dict[tuple[str, int], str] = {}

    def augment(slot: tuple[str, int], visited_pairs: set[str]) -> bool:
        route_id, _ = slot
        for record in eligible_by_route[route_id]:
            pair_id = record.pair_id
            if pair_id in visited_pairs:
                continue
            visited_pairs.add(pair_id)
            occupied_slot = pair_to_slot.get(pair_id)
            if occupied_slot is None or augment(occupied_slot, visited_pairs):
                pair_to_slot[pair_id] = slot
                slot_to_pair[slot] = pair_id
                return True
        return False

    for slot in slots:
        if not augment(slot, set()):
            raise AuditAllocationError(
                "balanced audit requires "
                f"{len(slots)} globally distinct matched generated pairs; "
                f"only {len(slot_to_pair)} can be allocated across the declared routes"
            )

    selected_pairs_by_route = {
        route_id: frozenset(
            slot_to_pair[(route_id, slot_index)]
            for slot_index in range(route_counts[route_id])
        )
        for route_id in remaining_routes
    }
    return tuple(
        record
        for route_id in remaining_routes
        for record in eligible_by_route[route_id]
        if record.pair_id in selected_pairs_by_route[route_id]
    )


def _validate_blinded_metrics(
    token_count: int,
    base_normalized_nll: float,
    automated_style_scores: tuple[tuple[str, float], ...],
) -> None:
    if type(token_count) is not int or token_count <= 0:
        raise ValueError("audit token_count must be positive")
    if (
        type(base_normalized_nll) not in (int, float)
        or not math.isfinite(base_normalized_nll)
        or base_normalized_nll < 0.0
    ):
        raise ValueError("audit base_normalized_nll must be finite and nonnegative")
    if type(automated_style_scores) is not tuple or not automated_style_scores:
        raise ValueError("audit automated_style_scores must be a nonempty tuple")
    if any(
        type(score) is not tuple
        or len(score) != 2
        or type(score[0]) is not str
        or not score[0].strip()
        or type(score[1]) not in (int, float)
        or not math.isfinite(score[1])
        for score in automated_style_scores
    ):
        raise ValueError("audit style scores must be finite named pairs")
    if len({name for name, _ in automated_style_scores}) != len(
        automated_style_scores
    ):
        raise ValueError("audit style score names must be unique")


def _require_text(value: str | None, name: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _require_sha256(value: str, name: str) -> None:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
