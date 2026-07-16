from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_normal_lm_imports_do_not_load_torch_or_transformers() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    source_path = str(repository_root / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (source_path, environment.get("PYTHONPATH", ""))
        if value
    )
    script = "\n".join(
        (
            "import sys",
            "import apm.lm",
            "import apm.lm.parity",
            "import apm.lm.generation",
            "assert 'torch' not in sys.modules",
            "assert 'transformers' not in sys.modules",
        )
    )

    completed = subprocess.run(
        (sys.executable, "-c", script),
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
