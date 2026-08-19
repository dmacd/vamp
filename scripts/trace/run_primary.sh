#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/workspace/vamp-trace/cache/huggingface
export HF_DATASETS_CACHE=/workspace/vamp-trace/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/workspace/vamp-trace/cache/huggingface/models

python -u -m apm.continual.trace.cli run --config configs/trace/primary.yaml
