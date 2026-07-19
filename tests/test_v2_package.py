from __future__ import annotations

import subprocess
import sys

from apm.data.text import tinyworlds_v2


def test_phase1_package_exposes_curated_cross_module_surface() -> None:
    expected = {
        "ArchiveSourceSelections",
        "AuditApprovalArtifact",
        "CANDIDATE_MODELS",
        "CostPreflight",
        "HttpxTransport",
        "OpenRouterClient",
        "Phase1ArtifactBuilder",
        "Phase1ReferenceInputs",
        "ReferenceProfile",
        "build_phase1_reference_inputs",
        "evaluate_route_quality",
        "score_tinystories_checkpoint_nll",
        "validate_phase1_reference",
    }

    assert expected.issubset(tinyworlds_v2.__all__)
    assert all(hasattr(tinyworlds_v2, name) for name in tinyworlds_v2.__all__)
    assert len(tinyworlds_v2.__all__) == len(set(tinyworlds_v2.__all__))
    assert "_surface_input" not in tinyworlds_v2.__all__


def test_package_import_does_not_import_optional_httpx() -> None:
    code = (
        "import sys; "
        "import apm.data.text.tinyworlds_v2; "
        "print(int(any(x == 'httpx' or x.startswith('httpx.') "
        "for x in sys.modules)))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "0"
