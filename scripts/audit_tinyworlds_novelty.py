"""Run the marked original-TinyStories lexical novelty gate for TinyWorlds v1."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile

from tqdm import tqdm

from apm.data.text.tinyworlds import load_tinyworlds_bundle
from apm.data.text.tinyworlds.novelty import (
    ORIGINAL_TINYSTORIES_REVISION,
    ORIGINAL_TINYSTORIES_TRAIN,
    audit_nonce_terms,
    novelty_terms_for_bundles,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TINYWORLDS_ROOT = REPOSITORY_ROOT / "data" / "tinyworlds" / "v1"
CORPUS_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "tinystories-original"
    / ORIGINAL_TINYSTORIES_TRAIN.filename
)
REPORT_PATH = TINYWORLDS_ROOT / "novelty_audit.json"


def main() -> None:
    """Stream the sole pinned corpus and persist a canonical zero-hit report."""
    temporary_directory = Path(tempfile.mkdtemp(prefix="tinyworlds-novelty-"))
    print(temporary_directory, flush=True)
    print("Phase 1/1: verify and stream the pinned original corpus", flush=True)
    bundles = tuple(
        load_tinyworlds_bundle(TINYWORLDS_ROOT / name)
        for name in ("calibration", "pilot")
    )
    with (
        tqdm(
            total=ORIGINAL_TINYSTORIES_TRAIN.size_bytes,
            desc="TinyWorlds novelty audit phase",
            unit="B",
            unit_scale=True,
        ) as phase_progress,
        tqdm(
            total=ORIGINAL_TINYSTORIES_TRAIN.size_bytes,
            desc="TinyWorlds novelty audit overall",
            unit="B",
            unit_scale=True,
        ) as overall_progress,
    ):
        def update_progress(byte_count: int) -> None:
            phase_progress.update(byte_count)
            overall_progress.update(byte_count)

        report = audit_nonce_terms(
            CORPUS_PATH,
            novelty_terms_for_bundles(bundles),
            progress_bytes=update_progress,
        )
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "revision": ORIGINAL_TINYSTORIES_REVISION,
                "corpus": asdict(ORIGINAL_TINYSTORIES_TRAIN),
                "audited_term_count": len(report.audited_terms),
                "terms": [asdict(term) for term in report.audited_terms],
                "hits": [
                    {
                        "term": asdict(hit.term),
                        "occurrence_count": hit.occurrence_count,
                    }
                    for hit in report.hits
                ],
                "passed": report.passed,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")
    if REPORT_PATH.is_file():
        if REPORT_PATH.read_bytes() != payload:
            raise RuntimeError(f"existing novelty report differs: {REPORT_PATH}")
    else:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = REPORT_PATH.with_name(f".{REPORT_PATH.name}.tmp")
        temporary_path.write_bytes(payload)
        temporary_path.replace(REPORT_PATH)
    if not report.passed:
        raise RuntimeError(f"TinyWorlds novelty audit found {len(report.hits)} hits")
    (temporary_directory / "completed.json").write_text(
        json.dumps(
            {
                "audited_term_count": len(report.audited_terms),
                "passed": True,
                "report_path": str(REPORT_PATH),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"novelty gate passed: {len(report.audited_terms)} terms, zero hits; "
        f"{REPORT_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
