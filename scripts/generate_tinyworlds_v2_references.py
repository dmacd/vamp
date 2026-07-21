"""Run the active Qwen/GPT-5.4-Mini TinyWorlds-v2 Phase 1 bakeoff."""

import os

# Package initialization can import JAX before the runner has a chance to
# configure it.  Set the allocator policy before importing any ``apm`` module.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from apm.data.text.tinyworlds_v2.two_route_bakeoff import main


if __name__ == "__main__":
    main()
