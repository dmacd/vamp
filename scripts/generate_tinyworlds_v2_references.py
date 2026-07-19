"""Generate the fixed TinyWorlds-v2 Phase 1 reference and bakeoff bundle."""

import os

# Package initialization can import JAX before the runner has a chance to
# configure it.  Set the allocator policy before importing any ``apm`` module.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from apm.data.text.tinyworlds_v2.phase1_runner import main


if __name__ == "__main__":
    main()
