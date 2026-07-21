"""Run the zero-cost exact-clause follow-up for the LoRA sidebar."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from apm.data.text.tinyworlds_v2.reasoning_sidebar_clause_probe import main


if __name__ == "__main__":
    main()
