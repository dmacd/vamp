"""Run the fixed Qwen/GPT TinyStories-LoRA learnability sidebar."""

import os

# JAX must see this before importing the runner through the package tree.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from apm.data.text.tinyworlds_v2.reasoning_sidebar_run import main


if __name__ == "__main__":
    main()
