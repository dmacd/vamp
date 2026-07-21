"""Run the active 20-by-two minimal-length-cue TinyWorlds-v2 bakeoff."""

import os

# Package initialization can import JAX before the runner configures it.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from apm.data.text.tinyworlds_v2.prompt_tuning import main


if __name__ == "__main__":
    main()
